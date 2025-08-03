# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch
import copy
import trimesh
import numpy as np
from PIL import Image
from typing import List
from DifferentiableRenderer.MeshRender import MeshRender
from utils.simplify_mesh_utils import remesh_mesh
from utils.multiview_utils import multiviewDiffusionNet
from utils.pipeline_utils import ViewProcessor
from utils.image_super_utils import imageSuperNet
from utils.uvwrap_utils import mesh_uv_wrap
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
import warnings
import gc
import time

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)


class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view, resolution):
        self.device = "cuda"

        self.multiview_cfg_path = "cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.max_selected_view_num = max_num_view
        self.resolution = resolution
        self.bake_exp = 4
        self.merge_method = "fast"

        # view selection
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.001, 0.001]

        # TODO(didiga): Remove this after testing
        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.4)

            self.candidate_camera_azims.append(-azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.4)


class Hunyuan3DPaintPipeline:

    def __init__(self, config=None) -> None:
        self.config = config if config is not None else Hunyuan3DPaintConfig()
        self.models = {}
        self.stats_logs = {}
        self.create_mesh_render()
        self.view_processor = ViewProcessor(self.config, self.render)
        self.load_models()

    def create_mesh_render(self):
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
        )
    def load_models(self):
        torch.cuda.empty_cache()
        self.models["super_model"] = imageSuperNet(self.config)
        self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        print("Models Loaded.")

    def set_active_pbr_settings(self, active_settings: List[str]):
        """Configure which PBR materials to use at runtime.
        
        Args:
            active_settings: List of materials to use (e.g., ["albedo"] for albedo-only)
        """
        if "multiview_model" in self.models:
            self.models["multiview_model"].set_active_pbr_settings(active_settings)
            print(f"Pipeline configured for PBR settings: {active_settings}")
        else:
            print("Warning: Multiview model not loaded, cannot set PBR settings")

    def get_active_pbr_settings(self):
        """Get currently active PBR settings."""
        if "multiview_model" in self.models:
            return self.models["multiview_model"].get_active_pbr_settings()
        return ["albedo", "mr"]  # Default

    def get_memory_usage_estimate(self):
        """Estimate VRAM usage for current PBR configuration."""
        active_settings = self.get_active_pbr_settings()
        base_usage = 8.0  # GB baseline
        per_material_usage = 2.0  # GB per additional material
        
        estimated_usage = base_usage + (len(active_settings) - 1) * per_material_usage
        return estimated_usage

    @torch.no_grad()
    def __call__(self, mesh_path=None, image_path=None, output_mesh_path=None, use_remesh=True, save_glb=True, learnable_shading_token=None, pbr_settings=None, posterize_color_list=None):
        """Generate texture for 3D mesh using multiview diffusion
        
        Args:
            pbr_settings: List of PBR materials to generate (e.g., ["albedo"] for albedo-only)
        """
        # Configure PBR settings if provided
        if pbr_settings is not None:
            self.set_active_pbr_settings(pbr_settings)

        if self.render is None:
            self.create_mesh_render()
            
        active_pbr_settings = self.get_active_pbr_settings()
        print(f"Generating textures for: {active_pbr_settings}")
        print(f"Estimated VRAM usage: {self.get_memory_usage_estimate():.1f} GB")

        # Ensure image_prompt is a list
        if isinstance(image_path, str):
            image_prompt = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image_prompt = image_path
        if not isinstance(image_prompt, List):
            image_prompt = [image_prompt]
        else:
            image_prompt = image_path

        # Process mesh
        path = os.path.dirname(mesh_path)
        if use_remesh:
            processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
            remesh_mesh(mesh_path, processed_mesh_path)
        else:
            processed_mesh_path = mesh_path

        # Output path
        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, f"textured_mesh.obj")

        # Load mesh
        mesh = trimesh.load(processed_mesh_path)
        mesh = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh=mesh)

        ########### View Selection #########
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )

        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)
        #position_map_pil = position_maps[0].cpu().numpy()
        #position_map_pil = (position_map_pil * 255).astype(np.uint8)
        #position_map_pil = Image.fromarray(position_map_pil)
        for i in range(len(normal_maps)):
            position_maps[i].save(os.path.join(path, f"positionmap_{i}.png"))
            normal_maps[i].save(os.path.join(path, f"normalmap_{i}.png"))

        ##########  Style  ###########
        image_caption = "high quality"
        image_style = []
        for image in image_prompt:
            image = image.resize((512, 512))
            if image.mode == "RGBA":
                white_bg = Image.new("RGB", image.size, (255, 255, 255))
                white_bg.paste(image, mask=image.getchannel("A"))
                image = white_bg
            image_style.append(image)
        image_style = [image.convert("RGB") for image in image_style]

        ###########  Multiview  ##########
        multiviews_pbr = self.models["multiview_model"](
            image_style,
            normal_maps + position_maps,
            prompt=image_caption,
            custom_view_size=self.config.resolution,
            resize_input=True,
            extra_shading_token=learnable_shading_token,
        )
        
        ###########  Enhance  ##########
        enhance_images = {}
        
        # Only process active PBR materials
        for material in active_pbr_settings:
            if material != "albedo":
                continue
            if material in multiviews_pbr:
                enhance_images[material] = copy.deepcopy(multiviews_pbr[material])
                if posterize_color_list is not None:
                    # Posterize colors if specified
                    quantize_start_time = time.time()
                    enhance_images[material] = quantize_texture(enhance_images[material], posterize_color_list)
                    quantize_end_time = time.time()
                    print(f"Posterization took {quantize_end_time - quantize_start_time:.2f} seconds")

                
                
                for i in range(len(enhance_images[material])):
                    enhance_images[material][i] = self.models["super_model"](enhance_images[material][i])
                    enhance_images[material][i].save(
                        os.path.join(path, f"enhanced_{material}_{i}.png")
                    )

        ###########  Bake  ##########
        # Resize enhanced images
        for material in enhance_images:
            if material != "albedo":
                continue
            for i in range(len(enhance_images[material])):
                enhance_images[material][i] = enhance_images[material][i].resize(
                    (self.config.render_size, self.config.render_size)
                )
        
        # Always bake albedo texture
        if "albedo" in enhance_images:
            texture, mask = self.view_processor.bake_from_multiview(
                enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )
            mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

                        
            ##########  inpaint  ###########
            texture = self.view_processor.texture_inpaint(texture, mask_np)


            self.render.set_texture(texture, force_set=True)
        
        # Conditionally bake metallic-roughness texture
        #if "mr" in enhance_images:
        #    texture_mr, mask_mr = self.view_processor.bake_from_multiview(
        #        enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        #    )
        #    mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        #    texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
        #    self.render.set_texture_mr(texture_mr)

        self.render.save_mesh(output_mesh_path, downsample=True)
        del texture

        if save_glb:
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))
            output_glb_path = output_mesh_path.replace(".obj", ".glb")
        gc.collect()
        torch.cuda.empty_cache()

        return output_mesh_path

def quantize_texture(texture, posterize_color_list):
    """Posterize texture colors based on a predefined color list."""
    if not posterize_color_list:
        return texture

    import numpy as np
    import torch
    from PIL import Image

    # Detect input type and get H×W×C float32 array in [0,255]
    is_tensor = isinstance(texture, torch.Tensor)
    is_pil = isinstance(texture, Image.Image)
    if is_tensor:
        device = texture.device
        tex_np = texture.detach().cpu().numpy()
    elif isinstance(texture, np.ndarray):
        tex_np = texture
    else:
        tex_np = np.asarray(texture)

    # Normalize to [0,255]
    if tex_np.dtype == np.uint8:
        tex255 = tex_np.astype(np.float32)
    else:
        tex255 = (tex_np * 255.0).astype(np.float32)

    h, w, c = tex255.shape
    palette = np.array(posterize_color_list, dtype=np.float32)  # (P,3)

    # Flatten and compute distances
    pixels = tex255.reshape(-1, c)                               # (N,3)
    diffs = pixels[:, None, :] - palette[None, :, :]             # (N,P,3)
    dist2 = np.sum(diffs * diffs, axis=2)                        # (N,P)
    idx = np.argmin(dist2, axis=1)                               # (N,)

    # Map to nearest palette color
    quant = palette[idx].reshape(h, w, c)                        # (H,W,3)
    quant_norm = quant / 255.0

    # Return same type as input
    if is_tensor:
        return torch.from_numpy(quant_norm).to(device)
    elif is_pil:
        return Image.fromarray(quant.astype(np.uint8))
    else:
        return quant_norm
        return quant_norm
