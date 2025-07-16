import torch
from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("encode a directory of .glb meshes to latents")
    p.add_argument("--input_dir", required=True, help="directory containing .glb files")
    p.add_argument("--output_dir", required=True, help="where to save latent .pt files")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae = ShapeVAE.from_pretrained(
        'tencent/Hunyuan3D-2.1',
        use_safetensors=False,
        variant='fp16',
        num_latents=4096,
    )

    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=0,
        num_uniform_points=81920,
    )
    for glb_path in sorted(in_dir.glob("*.glb")):
        surface = loader(str(glb_path)).to("cuda", dtype=torch.float16)
        latents = vae.encode(surface)
        out_file = out_dir / f"{glb_path.stem}_latents.pt"
        torch.save(latents, out_file)
        print(f"encoded {glb_path.name} → {out_file.name}")


if __name__ == "__main__":
    main()