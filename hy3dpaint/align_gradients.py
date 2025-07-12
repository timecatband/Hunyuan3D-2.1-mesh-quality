#!/usr/bin/env python3
"""
align_gradients.py  (rev-11 - final version with tolerant weld groups)
────────────────────────────────────────────────────────────────────────────
Align screen-space normal-map gradients with colour gradients of an existing
texture by optimising mesh vertices and/or UV coordinates.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer,
    MeshRenderer, SoftPhongShader, TexturesUV, TexturesVertex,
    look_at_view_transform, AmbientLights
)
import torchvision.transforms.functional as TF


# -------------------- Weld Group Helper --------------------
def get_weld_groups(mesh: trimesh.Trimesh) -> list:
    """
    Finds groups of vertex indices at the same 3D location,
    using tolerance for floating point inaccuracies.
    """
    # The 'digits' parameter sets the decimal precision for comparison.
    # A lower number (e.g., 5) is more tolerant than the default.
    return trimesh.grouping.group_rows(mesh.vertices, digits=5)

# -------------------- Laplacian Helpers --------------------
def _uniform_laplacian(mesh: Meshes, device: torch.device) -> torch.Tensor:
    faces = mesh.faces_packed()
    V = mesh.num_verts_per_mesh()[0].item()
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)
    idx = edges[:, 0] * V + edges[:, 1]
    unique = torch.unique(idx)
    e = torch.stack([unique // V, unique % V], dim=0)
    A = torch.sparse_coo_tensor(e, torch.ones(e.shape[1], device=device), (V, V))
    deg = torch.sparse.sum(A, dim=1).to_dense()
    D = torch.sparse_coo_tensor(torch.stack([torch.arange(V, device=device)] * 2), deg, (V, V))
    return D - A


def _cot_laplacian(mesh: Meshes, device: torch.device) -> torch.Tensor:
    try:
        from pytorch3d.ops.laplacian import cot_laplacian
        L, _ = cot_laplacian(mesh, norm_type='cot')
        return L.to(device)
    except Exception:
        pass
    logging.warning('Cotangent Laplacian unavailable, using uniform Laplacian')
    return _uniform_laplacian(mesh, device)


# -------------------- I/O Helpers --------------------
def _extract_tex_np(mat) -> np.ndarray:
    image_obj = None
    # Case 1: A simple material with a direct image
    if hasattr(mat, 'image') and mat.image is not None:
        image_obj = mat.image
    # Case 2: A PBR material where baseColorTexture is the image
    elif hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
        image_obj = mat.baseColorTexture  # <-- This handles the error

    if image_obj is None:
        raise RuntimeError('No colour texture found')

    # Convert the Pillow Image object to a numpy array
    arr = np.asarray(image_obj)
    return arr[..., :3] if arr.shape[-1] == 4 else arr

def load_glb(path: Path, device: torch.device) -> (Meshes, list):
    """Loads mesh data and identifies vertex groups at seams."""
    tm = trimesh.load(path, force='mesh')
    if isinstance(tm, trimesh.Scene):
        tm = tm.dump(concatenate=True)

    if not isinstance(tm, trimesh.Trimesh):
        raise RuntimeError('Expected single mesh in GLB')

    verts = torch.tensor(tm.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(tm.faces, dtype=torch.int64, device=device)

    # UVs and UV faces
    if tm.visual.kind == 'texture' and hasattr(tm.visual, 'uv') and tm.visual.uv is not None:
        uv = torch.tensor(tm.visual.uv, dtype=torch.float32, device=device)
        if hasattr(tm.visual, 'uv_faces') and tm.visual.uv_faces is not None:
            faces_uv = torch.tensor(tm.visual.uv_faces, dtype=torch.int64, device=device)
        else:
            faces_uv = faces.clone()
    else:
        logging.warning('No UVs found, using zero UVs.')
        uv = torch.zeros((len(verts), 2), dtype=torch.float32, device=device)
        faces_uv = faces.clone()

    # Texture map
    tex_np = None
    if tm.visual.kind == 'texture':
        try:
            tex_np = _extract_tex_np(tm.visual.material)
        except Exception as e:
            logging.warning(f'Texture load failed ({e}), using white')
    if tex_np is None:
        tex_np = np.ones((4, 4, 3), dtype=np.uint8) * 255

    tex = torch.tensor(tex_np, dtype=torch.float32, device=device) / 255.0
    tex = tex.flip(0)

    textures = TexturesUV(maps=tex.unsqueeze(0), faces_uvs=[faces_uv], verts_uvs=[uv])
    mesh_pytorch = Meshes(verts=[verts], faces=[faces], textures=textures)

    weld_groups = get_weld_groups(tm)
    return mesh_pytorch, weld_groups


def save_glb(mesh: Meshes, template: Path, out_path: Path):
    """Saves the optimized mesh, updating vertices and UVs."""
    m = mesh.detach().cpu()
    tm = trimesh.load(template, force='mesh')
    if isinstance(tm, trimesh.Scene):
        tm = tm.dump(concatenate=True)

    tm.vertices = m.verts_padded()[0].numpy()

    if m.textures is not None and hasattr(m.textures, 'verts_uvs_padded'):
        uvs = m.textures.verts_uvs_padded()
        if uvs is not None:
            tm.visual.uv = uvs[0].numpy()

    tm.export(out_path)


# -------------------- Optimisation --------------------

def optimise(
    mesh: Meshes,
    weld_groups: list,
    *, iters: int, lr: float,
    w_lap: float, w_disp: float,
    w_arap: float, w_flip: float,
    im_res: int, views: int,
    optimize_verts: bool, optimize_uvs: bool,
    log_every: int
) -> Meshes:
    device = mesh.device
    base_v = mesh.verts_padded()[0]
    faces = mesh.faces_packed()
    base_uv = mesh.textures.verts_uvs_padded()[0]
    faces_uv = mesh.textures.faces_uvs_padded()[0]
    maps = mesh.textures.maps_padded()

    delta = torch.zeros_like(base_v, requires_grad=optimize_verts)
    uv_delta = torch.zeros_like(base_uv, requires_grad=optimize_uvs)
    params = [x for x in (delta, uv_delta) if x.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    # ARAP edges
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    e = torch.unique(torch.sort(e, dim=1)[0], dim=0)
    v0, v1 = base_v[e[:, 0]], base_v[e[:, 1]]
    l0 = (v0 - v1).norm(dim=1)

    # Face normals
    f0 = base_v[faces[:, 1]] - base_v[faces[:, 0]]
    f1 = base_v[faces[:, 2]] - base_v[faces[:, 0]]
    n0 = F.normalize(torch.cross(f0, f1, dim=1), dim=1)

    # Camera
    center = base_v.mean(0, keepdim=True)
    radius = (base_v - center).norm(dim=1).max().item() * 1.8
    elev = torch.full((views,), 15.0, device=device)
    azim = torch.linspace(-180, 180, views, device=device)
    R, T = look_at_view_transform(dist=radius, elev=elev, azim=azim, device=device)
    cam = FoVPerspectiveCameras(R=R, T=T, device=device)
    raster = MeshRasterizer(cameras=cam,
        raster_settings=RasterizationSettings(image_size=im_res, faces_per_pixel=1))

    # Renderers
    lit_renderer = MeshRenderer(raster, SoftPhongShader(device=device, cameras=cam))
    ambient = AmbientLights(device=device)
    tex_renderer = MeshRenderer(raster, SoftPhongShader(device=device, cameras=cam, lights=ambient))

    # Sobel filters for gradient calculation
    sob = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 8
    sx, sy = sob, sob.permute(0, 1, 3, 2) * -1
    L = _cot_laplacian(mesh, device)

    for it in tqdm(range(iters), desc='Optimising', unit='it'):
        verts = base_v + delta
        uvs = base_uv + uv_delta

        # Build mesh
        tex = TexturesUV(maps=maps, faces_uvs=[faces_uv], verts_uvs=[uvs])
        m = Meshes(verts=[verts], faces=[faces], textures=tex)

        # Render texture-driven RGB
        rgb = tex_renderer(m, cameras=cam)[..., :3]

        # Render normals
        ncol = m.verts_normals_padded()[0] * 0.5 + 0.5
        mn = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex([ncol]))
        nrm = lit_renderer(mn, cameras=cam)[..., :3]

        # Gradients
        gn = F.normalize(torch.stack([F.conv2d(TF.rgb_to_grayscale(nrm.permute(0, 3, 1, 2)), sx, padding=1),
                                      F.conv2d(TF.rgb_to_grayscale(nrm.permute(0, 3, 1, 2)), sy, padding=1)], -1), dim=-1)
        gt_raw = torch.stack([F.conv2d(TF.rgb_to_grayscale(rgb.permute(0, 3, 1, 2)), sx, padding=1),
                              F.conv2d(TF.rgb_to_grayscale(rgb.permute(0, 3, 1, 2)), sy, padding=1)], -1)
        # Loss based on angular difference, not magnitude
        gt = F.normalize(gt_raw, dim=-1)
        E_align = (1 - (gn * gt).abs().sum(-1)).mean()

        # Regularization terms
        lap = torch.sparse.mm(L, delta)
        E_lap = (lap ** 2).sum()
        E_disp = (delta ** 2).sum()
        v0n, v1n = verts[e[:, 0]], verts[e[:, 1]]
        E_arap = ((v0n - v1n).norm(dim=1) - l0).pow(2).mean()
        f0n = verts[faces[:, 1]] - verts[faces[:, 0]]
        f1n = verts[faces[:, 2]] - verts[faces[:, 0]]
        n1 = F.normalize(torch.cross(f0n, f1n, dim=1), dim=1)
        E_flip = F.relu(-torch.sum(n0 * n1, dim=1)).mean()

        loss = E_align + w_lap * E_lap + w_disp * E_disp + w_arap * E_arap + w_flip * E_flip

        opt.zero_grad()
        loss.backward()
        opt.step()

        # --- THIS IS THE NEW, ROBUST FIX FOR CRACKS ---
        # After the optimizer step, force vertices in weld groups to the same position.
        if optimize_verts:
            with torch.no_grad():
                for group in weld_groups:
                    if isinstance(group, (list, np.ndarray)) and len(group) > 1:
                        # Calculate the average position delta for the group
                        avg_delta = delta[group].mean(dim=0, keepdim=True)
                        # Enforce that all vertices in the group have the same delta
                        delta[group] = avg_delta
        # --- END OF FIX ---

        # Project vertex movement to tangent plane
        if optimize_verts:
            with torch.no_grad():
                normals = mn.verts_normals_padded()[0]
                comp = (delta * normals).sum(dim=1, keepdim=True)
                delta.sub_(normals * comp)

        # UV smoothing
        if optimize_uvs:
            with torch.no_grad():
                uv_delta.sub_(torch.sparse.mm(L, uv_delta) * 0.1 * w_lap)

        if it % 5 == 0 or it == iters - 1:
            print(f"[{it}/{iters}] loss {loss:.4g} align {E_align:.4g}")

    return Meshes(verts=[base_v + delta.detach()], faces=[faces],
                  textures=TexturesUV(maps=maps, faces_uvs=[faces_uv], verts_uvs=[base_uv + uv_delta.detach()]))

# -------------------- CLI --------------------
def _cli():
    p = argparse.ArgumentParser(description='Align screen-space gradients by optimizing mesh vertices and/or UVs.')
    p.add_argument('--input', type=Path, required=True, help='Input mesh file (e.g., mesh.glb)')
    p.add_argument('--output', type=Path, required=True, help='Output mesh file (e.g., mesh_optimized.glb)')
    p.add_argument('--iters', type=int, default=600, help='Number of optimization iterations')
    p.add_argument('--image_size', type=int, default=512, help='Render resolution for gradient calculation')
    p.add_argument('--views', type=int, default=1, help='Number of camera views to optimize over')
    p.add_argument('--lr', type=float, default=2e-3, help='Learning rate for the optimizer')
    p.add_argument('--lap', type=float, default=1e-3, help='Laplacian regularization weight')
    p.add_argument('--disp', type=float, default=1e-4, help='Displacement regularization weight')
    p.add_argument('--arap', type=float, default=0.0, help='ARAP regularization weight')
    p.add_argument('--flip', type=float, default=0.0, help='Face-flipping penalty weight')
    p.add_argument('--opt-uvs', action='store_true', help='Enable optimization of UV coordinates')
    p.add_argument('--no-verts', action='store_true', help='Disable optimization of vertex positions')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Computation device')
    p.add_argument('--verbose', action='store_true', help='Enable detailed logging')
    return p.parse_args()


def main():
    args = _cli()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(asctime)s %(levelname)s: %(message)s')
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    mesh, weld_groups = load_glb(args.input, dev)

    final = optimise(
        mesh,
        weld_groups=weld_groups,
        iters=args.iters, lr=args.lr,
        w_lap=args.lap, w_disp=args.disp,
        w_arap=args.arap, w_flip=args.flip,
        im_res=args.image_size, views=args.views,
        optimize_verts=not args.no_verts,
        optimize_uvs=args.opt_uvs,
        log_every=50
    )
    save_glb(final, args.input, args.output)
    logging.info('Saved → %s', args.output)


if __name__ == '__main__':
    main()