import os
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss
from peft import LoraConfig, get_peft_model
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler

class LatentDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pt')]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        return torch.load(self.files[idx]).squeeze(0)

def collate_latents(batch):
    return torch.stack(batch, dim=0)

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
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--epochs",        type=int, default=10)
    parser.add_argument("--timesteps",     type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max norm for gradient clipping")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="number of gradient accumulation steps")
    parser.add_argument("--save_every_epochs", type=int, default=1,
                        help="save adapters every N epochs")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="path to a saved checkpoint directory (e.g. .../epoch_5) to resume from")
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device
    dtype  = getattr(torch, args.dtype)

    # 1) load pipeline & base DiT model
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        octree_resolution=128,
        device=device,
        dtype=dtype
    )
    base_dit = pipe.model

    # 2) resume or new LoRA setup
    if args.resume_checkpoint:
        import re
        from peft import PeftModel
        # extract last completed epoch from folder name
        m = re.search(r'epoch_(\d+)', os.path.basename(args.resume_checkpoint))
        start_epoch = int(m.group(1)) if m else 0
        # load saved LoRA adapters
        dit_model = PeftModel.from_pretrained(base_dit, args.resume_checkpoint, is_trainable=True).to(device)
        dit_model.train()
        # load the learnable cond token
        raw_tok = torch.load(os.path.join(args.resume_checkpoint, "cond_token.pt"), map_location=device)
        cond_tok = nn.Parameter(raw_tok.to(device=device, dtype=dtype))
    else:
        start_epoch = 0
        # 2) apply LoRA
        lora_cfg = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            target_modules="all-linear" #args.target_modules
        )
        dit_model = get_peft_model(base_dit, lora_cfg).to(device)
        dit_model.train()
        # 3) single learnable conditional token
        cond_tok = nn.Parameter(torch.randn(1, 1, args.cond_dim, device=device, dtype=dtype))

    # 4) optimizer & loss
    optimizer = torch.optim.AdamW([*dit_model.parameters(), cond_tok], lr=args.lr)
    loss_fn   = MSELoss()

    # 5) scheduler
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=args.timesteps)

    # 6) data loader
    ds = LatentDataset(args.latents_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_latents)

    optimizer.zero_grad()
    # 7) training loop
    for epoch in range(start_epoch, args.epochs):
        # initialize tracking for this epoch
        sum_loss = 0.0
        num_steps = 0

        for step, latents in enumerate(dl):
            latents = latents.to(device=device, dtype=torch.float32)
            
            # Sample random timestep (sigma) from [0, 1] for flow matching
            t = torch.rand(latents.size(0), device=device)
            
            # Flow matching: interpolate between noise and data
            noise = torch.randn_like(latents)
            # x_t = (1 - t) * noise + t * latents  # Standard flow matching interpolation
            # But looking at scheduler, it uses: x_t = sigma * noise + (1 - sigma) * latents
            sigma = t.view(-1, *([1] * (len(latents.shape) - 1)))  # Reshape for broadcasting
            x_t = sigma * noise + (1.0 - sigma) * latents
            
            # Convert t to timesteps for model input (scheduler expects timesteps 0-1000)
            timesteps = t * args.timesteps

            cond_tok = cond_tok.repeat(latents.size(0), 1, 1)  # Expand to match batch size
            cond = {"main": cond_tok}

            pred = dit_model(x_t.to(torch.float32), timesteps.to(torch.float32)/args.timesteps, cond)

            # Flow matching target: the velocity field v_t = x_1 - x_0 = latents - noise
            target = latents - noise
            loss = loss_fn(pred, target) / args.grad_accum_steps
            loss.backward()

            # update running stats
            sum_loss += loss.item()
            num_steps += 1

            # step optimizer once every grad_accum_steps
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1 == len(dl)):
                total_norm = torch.nn.utils.clip_grad_norm_(
                    list(dit_model.parameters()) + [cond_tok],
                    args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()

                # print running average
                avg_loss = sum_loss / num_steps
                print(f"Step {step+1}/{len(dl)} — avg loss: {avg_loss:.4f}, grad norm: {total_norm:.4f}")

        # epoch summary
        avg_epoch_loss = sum_loss / num_steps if num_steps > 0 else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} — avg loss: {avg_epoch_loss:.4f}")

        # checkpoint every N epochs
        if (epoch + 1) % args.save_every_epochs == 0:
            ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            dit_model.save_pretrained(ckpt_dir)
            torch.save(cond_tok, os.path.join(ckpt_dir, "cond_token.pt"))
            print(f"Saved adapters at epoch {epoch+1} → {ckpt_dir}")

    # 8) final save
    os.makedirs(args.output_dir, exist_ok=True)
    dit_model.save_pretrained(args.output_dir)
    # Save the learnable token
    torch.save(cond_tok, os.path.join(args.output_dir, "cond_token.pt"))

if __name__ == "__main__":
    main()
