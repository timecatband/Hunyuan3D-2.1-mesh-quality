import os
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from peft import LoraConfig, get_peft_model, PeftModel   # added PeftModel
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
import hy3dshape.models.diffusion.transport

class LatentDataset(Dataset):
    def __init__(self, folder):
        self.samples = []
        self.remover = BackgroundRemover()
        # find .pt files and their matching image
        for f in os.listdir(folder):
            if not f.endswith('.pt'):
                continue
            latent_path = os.path.join(folder, f)
            base, _ = os.path.splitext(latent_path)
            img_path = None
            base = base.replace('_latents', '')
            for ext in ('.jpg','.jpeg','.png'):
                cand = base + ext
                if os.path.exists(cand):
                    img_path = cand
                    break
            if img_path is None:
                raise FileNotFoundError(f"No image for {latent_path}")
            self.samples.append((latent_path, img_path))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        latent_path, img_path = self.samples[idx]
        latent = torch.load(latent_path).squeeze(0)
        # load and strip background once
        img = Image.open(img_path).convert("RGBA")
        img = self.remover(img)
        return latent, img

def collate_latents(batch):
    latents, images = zip(*batch)
    return torch.stack(latents, dim=0), list(images)

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA training for Flow-Matching DiT")
    parser.add_argument("--model_path",    type=str, required=False, default="tencent/Hunyuan3D-2.1", help="pretrained DiT model identifier or path")
    parser.add_argument("--latents_dir",   type=str, required=True, help="folder of .pt latent files")
    parser.add_argument("--output_dir",    type=str, default="lora_adapters", help="where to save LoRA adapters")
    parser.add_argument("--device",        type=str, default="cuda")
    parser.add_argument("--dtype",         type=str, default="float32", choices=["float16","float32"])
    parser.add_argument("--r",             type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha",         type=int, default=16, help="LoRA alpha")
    parser.add_argument("--target_modules",nargs="+", default=["to_q","to_v"])
    parser.add_argument("--cond_dim",      type=int, default=1024, help="dimension of the learnable token")
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--lr",            type=float, default=4e-5)
    parser.add_argument("--epochs",        type=int, default=10)
    parser.add_argument("--timesteps",     type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max norm for gradient clipping")
    parser.add_argument("--grad_accum_steps",   type=int,   default=1, help="number of gradient accumulation steps")
    parser.add_argument("--save_every_epochs",  type=int,   default=1, help="save adapters every N epochs")
    parser.add_argument("--resume_checkpoint",  type=str,   default=None, help="path to a saved checkpoint (e.g. .../epoch_5)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device
    dtype  = getattr(torch, args.dtype)

    # 1) load pipeline & base DiT model
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path, octree_resolution=128, device=device, dtype=dtype
    )
    base_dit = pipe.model

    # 2) resume or new LoRA + cond token
    if args.resume_checkpoint:
        import re
        m = re.search(r'epoch_(\d+)', os.path.basename(args.resume_checkpoint))
        start_epoch = int(m.group(1)) if m else 0
        dit_model = PeftModel.from_pretrained(base_dit, args.resume_checkpoint, is_trainable=True).to(device)
        dit_model.train()
        raw_tok = torch.load(os.path.join(args.resume_checkpoint, "cond_token.pt"), map_location=device)
        cond_tok = nn.Parameter(raw_tok.to(device=device, dtype=dtype))
    else:
        start_epoch = 0
        lora_cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, target_modules="all-linear")
        dit_model = get_peft_model(base_dit, lora_cfg).to(device)
        dit_model.train()
        cond_tok = nn.Parameter(torch.randn(1, 1, args.cond_dim, device=device, dtype=dtype))

    # 3) optimizer, loss, scheduler, dataloader
    optimizer = torch.optim.AdamW([*dit_model.parameters(), cond_tok], lr=args.lr)
    loss_fn   = MSELoss()
    sched     = FlowMatchEulerDiscreteScheduler(num_train_timesteps=args.timesteps)
    ds        = LatentDataset(args.latents_dir)
    dl        = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_latents)

    optimizer.zero_grad()

    transport = hy3dshape.models.diffusion.transport.create_transport()
    z_scale_factor = 1.0039506158752403


    # 4) training loop with grad‐accum & checkpointing
    for epoch in range(start_epoch, args.epochs):
        sum_loss = 0.0
        num_steps = 0

        for step, (latents, images) in enumerate(dl):
            latents = latents.to(device=device, dtype=torch.float32)
            latents = z_scale_factor * latents  # scale latents

            # prepare conditioning
            cond_inputs = pipe.prepare_image(images, None)
            img_tensor  = cond_inputs.pop("image")
            cond = pipe.encode_cond(
                image=img_tensor.to(device=device, dtype=dtype),
                additional_cond_inputs=cond_inputs,
                do_classifier_free_guidance=False,
                dual_guidance=False
            )

            # append learnable token + dropout
            bsz = latents.size(0)
            tok = cond_tok.expand(bsz, -1, -1)
            p_drop_both  = 0.05
            p_drop_image = 0.10
            u = torch.rand(1).item()
            if u < p_drop_both:
                cond["main"] = torch.zeros_like(cond["main"])
            elif u < p_drop_both + p_drop_image:
                cond["main"] = tok
            else:
                cond["main"] = cond["main"] + tok

            loss = transport.training_losses(
                dit_model, latents, dict(contexts=cond)
            )["loss"]#.mean()
            loss.backward()

            sum_loss += loss.item()
            num_steps += 1

            # optimizer step
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1 == len(dl)):
                total_norm = torch.nn.utils.clip_grad_norm_(
                    list(dit_model.parameters()) + [cond_tok], args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                avg_loss = sum_loss / num_steps
                print(f"Step {step+1}/{len(dl)} — avg loss: {avg_loss:.4f}, grad norm: {total_norm:.4f}")

        # epoch summary
        avg_epoch_loss = sum_loss / num_steps if num_steps > 0 else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} — avg loss: {avg_epoch_loss:.4f}")

        # checkpoint
        if (epoch + 1) % args.save_every_epochs == 0:
            ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            dit_model.save_pretrained(ckpt_dir)
            torch.save(cond_tok, os.path.join(ckpt_dir, "cond_token.pt"))
            print(f"Saved adapters at epoch {epoch+1} → {ckpt_dir}")

    # 5) final save
    os.makedirs(args.output_dir, exist_ok=True)
    dit_model.save_pretrained(args.output_dir)
    torch.save(cond_tok, os.path.join(args.output_dir, "cond_token.pt"))

if __name__ == "__main__":
    main()
