# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

try:
    from utils.torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
import sys
from hunyuanpaintpbr.unet.model import HunyuanPaint
import peft
from src.data.dataloader.objaverse_loader_forTexturePBR import TextureDataset
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

SD_TARGET_MODULES = [
    # Standard attention modules
    "to_q", "to_k", "to_v", "to_out.0",
    
    # MR-specific attention modules (CRITICAL - these were missing!)
    "to_q_mr", "to_k_mr", "to_v_mr", "to_out_mr.0",
    
    # Multiview attention modules (CRITICAL - these were missing!)
    "attn_multiview.to_q", "attn_multiview.to_k", "attn_multiview.to_v",
    
    # Reference view attention modules (CRITICAL - these were missing!)
    "attn_refview.to_q", "attn_refview.to_k", "attn_refview.to_v",
    "attn_refview.processor.to_v_mr",  # This is MR-specific!
    
    # DINO attention modules (CRITICAL - these were missing!)
    "attn_dino.to_q", "attn_dino.to_k", "attn_dino.to_v",
    
    # Processor modules
    "processor.to_q_mr", "processor.to_k_mr", "processor.to_v_mr",
    
    # Projection modules
    "proj_in", "proj_out", "proj",
    
    # Feed-forward networks
    "ff.net.0.proj", "ff.net.2",
    
    # Convolution modules
    "conv", "conv1", "conv2", "conv_shortcut",
    
    # Time embedding
    "time_emb_proj",
    
    # DINO projection (if present)
    "image_proj_model_dino.proj"
]

def load_pretrained_pipeline(max_num_view: int, resolution: int):
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    pipe = Hunyuan3DPaintPipeline(conf)
    pipe.models["super_model"] = None
    return pipe

def convert_to_training_module(paint_pipeline):
    pipeline = paint_pipeline.models["multiview_model"].pipeline
    pipeline = HunyuanPaint(pretrained_pipeline=pipeline,
                            view_size=512,
                            with_normal_map=True,
                            with_position_map=True)
    del paint_pipeline
    return pipeline

def setup_lora(pipeline):
    lora_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=SD_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )
    pipeline.unet = peft.get_peft_model(pipeline.unet, lora_config)
    print("LORA configuration applied to UNet")

def load_dataset(json_path):
    dataset = TextureDataset(json_path=json_path, num_view=6, image_size=512, lighting_suffix_pool=["light_PL", "light_AL", "light_ENVMAP", "light_KEY"])
    print("Dataset loaded with length:", len(dataset))
    return dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path",    type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default="outputs")
    parser.add_argument("--num_epochs",   type=int, default=5)
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--save_every",    type=int, default=1)
    parser.add_argument("--accum_steps",   type=int, default=1)
    parser.add_argument("--resolution",    type=int, default=512,
                        help="training image resolution, e.g. 256/320/512")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    paint_pipeline = load_pretrained_pipeline(max_num_view=6,
                                              resolution=args.resolution)
    train_module   = convert_to_training_module(paint_pipeline)
    setup_lora(train_module)
    dataset        = load_dataset(args.json_path)
    dataloader     = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # move model and all pipeline parts to device
    train_module.to(device)
    train_module.pipeline.vae.to(device)
    train_module.pipeline.vae.enable_slicing()
    train_module.pipeline.unet.to(device)
    train_module.pipeline.text_encoder.to(device)
    if hasattr(train_module, "dino_v2"):
        train_module.dino_v2.to(device)
    train_module.dino_v2.eval()

    # Quick memory optimizations
    # 1) Enable xFormers memory‐efficient attention & attention slicing
    #if hasattr(train_module.pipeline.unet, "enable_xformers_memory_efficient_attention"):
        #train_module.pipeline.unet.enable_xformers_memory_efficient_attention()
    if hasattr(train_module.pipeline.unet, "enable_attention_slicing"):
        train_module.pipeline.unet.enable_attention_slicing()
    # 2) Cast major components to FP16
    #train_module.unet.half()
    #train_module.pipeline.vae.half()
    #train_module.pipeline.text_encoder.half()
    #if hasattr(train_module, "dino_v2"):
    #    train_module.dino_v2.half()

    enable_gradient_checkpointing = True  # Set to True to enable gradient checkpointing
    if enable_gradient_checkpointing:
        train_module.pipeline.unet.enable_gradient_checkpointing()
        train_module.pipeline.vae.enable_gradient_checkpointing()
        # also enable gradient checkpointing on text encoder to save memory
        if hasattr(train_module.pipeline.text_encoder, "enable_gradient_checkpointing"):
            train_module.pipeline.text_encoder.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled for UNet")
    train_module.pipeline.text_encoder.cpu()

    # freeze all modules except UNet to avoid storing extra activations
    for name, param in train_module.named_parameters():
        if "unet" not in name:
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(train_module.unet.parameters(), lr=args.lr)
    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in train_module.unet.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    # Add AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()

    gc.collect()
    torch.cuda.empty_cache()

    for epoch in range(args.num_epochs):
        train_module.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
        total_loss = 0.0
        for step, batch in pbar:
            # Use autocast for mixed precision
            with torch.cuda.amp.autocast():
                loss = train_module.training_step(batch, step, supervise_mr=False)
                loss = loss / args.accum_steps
            
            # Scale loss and backward
            scaler.scale(loss).backward()
            total_loss += loss.item() * args.accum_steps

            # step optimizer every accum_steps
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(dataloader):
                # compute grad norm with scaler
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    train_module.unet.parameters(), max_norm=1e9
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                avg_step_loss = total_loss / ((step + 1) / args.accum_steps)
                pbar.set_postfix({"loss": f"{avg_step_loss:.4f}", "grad_norm": f"{grad_norm:.4f}"})

        epoch_avg = total_loss / len(dataloader)
        print(f"Epoch {epoch} | avg_loss: {epoch_avg:.4f}")

        # free up any leftover GPU memory
        torch.cuda.empty_cache()

        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"lora_epoch_{epoch+1}")
            train_module.unet.save_pretrained(ckpt_dir)
            print(f"Saved LoRA checkpoint to {ckpt_dir}")

if __name__ == "__main__":
    main()








