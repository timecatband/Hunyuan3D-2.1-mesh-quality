import os
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from peft import LoraConfig, get_peft_model
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler

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

    # 2) apply LoRA
    lora_cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, target_modules=args.target_modules)
    dit_model = get_peft_model(base_dit, lora_cfg).to(device)
    dit_model.train()

    # single learnable conditional token
    cond_tok = nn.Parameter(torch.randn(1, 1, args.cond_dim, device=device, dtype=dtype))

    # 3) optimizer & loss
    optimizer = torch.optim.AdamW([*dit_model.parameters(), cond_tok], lr=args.lr)
    loss_fn   = MSELoss()

    # 4) scheduler
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=args.timesteps)

    # 5) data loader
    ds = LatentDataset(args.latents_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_latents)

    # 6) training loop
    for epoch in range(args.epochs):
        for latents, images in dl:
            latents = latents.to(device=device, dtype=torch.float32)
            # prepare conditioning from preprocessed images
            cond_inputs = pipe.prepare_image(images, None)
            img_tensor = cond_inputs.pop('image')
            cond = pipe.encode_cond(
                image=img_tensor.to(device=device, dtype=dtype),
                additional_cond_inputs = cond_inputs,
                do_classifier_free_guidance=False,
                dual_guidance=False
            )

            # append learnable token to image conditioning
            bsz = latents.size(0)
            tok = cond_tok.expand(bsz, -1, -1)

            p_cond_dropout = 0.3
            if torch.rand(1).item() < p_cond_dropout:
                cond["main"] = tok
            else:
                cond["main"] = torch.cat([cond["main"], tok], dim=1)

            # sample and mix
            t = torch.rand(latents.size(0), device=device)
            noise = torch.randn_like(latents)
            sigma = t.view(-1, *([1] * (latents.ndim - 1)))
            x_t = sigma * noise + (1.0 - sigma) * latents
            timesteps = t * args.timesteps

            # predict
            pred = dit_model(x_t, timesteps, cond)

            target = latents - noise
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dit_model.parameters(), args.max_grad_norm)
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.epochs} — loss {loss.item():.4f}")

    # 7) save adapters
    os.makedirs(args.output_dir, exist_ok=True)
    dit_model.save_pretrained(args.output_dir)
    # save learned conditional token
    torch.save(cond_tok, os.path.join(args.output_dir, "cond_token.pt"))

if __name__ == "__main__":
    main()
