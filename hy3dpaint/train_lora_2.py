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
from peft import PeftModel  # Add import for resuming LoRA
from src.data.dataloader.objaverse_loader_forTexturePBR import TextureDataset
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

# Add Prodigy optimizer support
try:
    from prodigyopt import Prodigy  # type: ignore
    PRODIGY_AVAILABLE = True
except ImportError:
    PRODIGY_AVAILABLE = False

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

def setup_lora(pipeline, lora_rank):
    lora_config = peft.LoraConfig(
        r=lora_rank,
        lora_alpha=16,
        target_modules=SD_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )
    pipeline.unet = peft.get_peft_model(pipeline.unet, lora_config)
    print(f"LORA configuration applied with rank {lora_rank}")

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
    # Prodigy optimizer options
    parser.add_argument("--use_prodigy", action="store_true",
                        help="Use Prodigy optimizer for training")
    parser.add_argument("--prodigy_d_coef", type=float, default=1.0,
                        help="Prodigy D coefficient for optimizer")
    # LoRA training options
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank for adapter")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to LoRA checkpoint directory to resume from")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    paint_pipeline = load_pretrained_pipeline(max_num_view=6,
                                              resolution=args.resolution)
    train_module   = convert_to_training_module(paint_pipeline)
    if not args.resume_from:
        setup_lora(train_module, args.lora_rank)
    # Resume from existing LoRA checkpoint if provided
    if args.resume_from:
        print(f"Resuming LoRA weights from {args.resume_from}")
        train_module.unet = PeftModel.from_pretrained(train_module.unet, args.resume_from, is_trainable=True)
        train_module.unet.train()

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
    #if hasattr(train_module.pipeline.unet, "enable_attention_slicing"):
    #    train_module.pipeline.unet.enable_attention_slicing()
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

    # Create learnable albedo token parameter
    learnable_albedo_token = torch.nn.Parameter(torch.randn(1, 77, 1024, device=device) * 0.01)
    
    # Load learnable token if resuming from checkpoint
    if args.resume_from:
        token_path = os.path.join(args.resume_from, "learnable_albedo_token.pt")
        if os.path.exists(token_path):
            learnable_albedo_token.data = torch.load(token_path, map_location=device)
            print(f"Loaded learnable albedo token from {token_path}")
        else:
            print("Warning: No learnable albedo token found in checkpoint, using random initialization")

    # Setup optimizer - include both UNet parameters and learnable token
    unet_params = list(train_module.unet.parameters())
    all_params = unet_params + [learnable_albedo_token]
    
    if args.use_prodigy:
        if not PRODIGY_AVAILABLE:
            raise ImportError("Prodigy optimizer is required but not available. Install with: pip install prodigyopt")
        print(f"Using Prodigy optimizer with D coefficient: {args.prodigy_d_coef}")
        optimizer = Prodigy(
            all_params,
            lr=1.0,
            d_coef=args.prodigy_d_coef,
            betas=(0.9, 0.99),
            beta3=None,
            weight_decay=1e-2,
            eps=1e-8,
            use_bias_correction=True,
            safeguard_warmup=True
        )
    else:
        print(f"Using AdamW optimizer with learning rate: {args.lr}")
        optimizer = torch.optim.AdamW(all_params, lr=args.lr)

    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in train_module.unet.parameters() if p.requires_grad)
    trainable_params += learnable_albedo_token.numel()
    print(f"Number of trainable parameters: {trainable_params}")
    # Get learned_text_clip
    learned_clip = getattr(train_module.unet, "learned_text_clip", None)
    print("Learned clip requires grad:", learned_clip.requires_grad if learned_clip else "N/A")
    print("LEarned clip stats:", 
          f"shape: {learned_clip.shape if learned_clip is not None else 'N/A'}, " 
          f"requires_grad: {learned_clip.requires_grad if learned_clip is not None else 'N/A'}" 
          f"mean: {learned_clip.mean().item() if learned_clip is not None else 'N/A'}, "
          f"std: {learned_clip.std().item() if learned_clip is not None else 'N/A'}")
    
    # Print learnable albedo token stats
    print(f"Learnable albedo token stats: shape: {learnable_albedo_token.shape}, "
          f"mean: {learnable_albedo_token.mean().item():.6f}, std: {learnable_albedo_token.std().item():.6f}")

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
                loss = train_module.training_step(batch, step, supervise_mr=False, learnable_albedo_token=learnable_albedo_token)
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
                
                # Print learnable parameter stats every 10 steps
                if (step + 1) % 10 == 0:
                    token_mean = learnable_albedo_token.mean().item()
                    token_std = learnable_albedo_token.std().item()
                    print(f"Step {step+1}: Learnable token mean={token_mean:.6f}, std={token_std:.6f}")
                
                pbar.set_postfix({"loss": f"{avg_step_loss:.4f}", "grad_norm": f"{grad_norm:.4f}"})

        epoch_avg = total_loss / len(dataloader)
        print(f"Epoch {epoch} | avg_loss: {epoch_avg:.4f}")

        # free up any leftover GPU memory
        torch.cuda.empty_cache()

        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"lora_epoch_{epoch+1}")
            train_module.unet.save_pretrained(ckpt_dir)
            # Save learnable albedo token
            token_path = os.path.join(ckpt_dir, "learnable_albedo_token.pt")
            torch.save(learnable_albedo_token.data, token_path)
            print(f"Saved LoRA checkpoint to {ckpt_dir}")
            print(f"Saved learnable albedo token to {token_path}")

if __name__ == "__main__":
    main()








