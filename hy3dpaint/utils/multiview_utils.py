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

import os
import torch
import random
import numpy as np
from PIL import Image
from typing import List
import huggingface_hub
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        self.device = config.device

        cfg_path = config.multiview_cfg_path
        custom_pipeline = config.custom_pipeline
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        model_path = huggingface_hub.snapshot_download(
            repo_id=config.multiview_pretrained_path,
            allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
        )

        model_path = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=custom_pipeline, 
            torch_dtype=torch.float16
        )

        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        pipeline.set_progress_bar_config(disable=True)
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        self.pipeline = pipeline.to(self.device)

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            self.dino_v2 = self.dino_v2.to(self.device)

    def set_active_pbr_settings(self, active_settings: List[str]):
        """Configure which PBR materials to use at runtime.
        
        Args:
            active_settings: List of materials to use (e.g., ["albedo"] for albedo-only)
        """
        if hasattr(self.pipeline, 'set_active_pbr_settings'):
            self.pipeline.set_active_pbr_settings(active_settings)
            print(f"Multiview model configured for PBR settings: {active_settings}")
        else:
            print("Warning: Pipeline does not support dynamic PBR settings")

    def get_active_pbr_settings(self):
        """Get currently active PBR settings."""
        if hasattr(self.pipeline, 'get_active_pbr_count'):
            # Try to get from UNet if available
            if hasattr(self.pipeline.unet, 'active_pbr_setting'):
                return self.pipeline.unet.active_pbr_setting
            elif hasattr(self.pipeline.unet, 'pbr_setting'):
                return self.pipeline.unet.pbr_setting
        return ["albedo", "mr"]  # Default fallback

    def get_memory_usage_estimate(self):
        """Estimate VRAM usage for current PBR configuration."""
        active_settings = self.get_active_pbr_settings()
        base_usage = 6.0  # GB baseline for multiview model
        per_material_usage = 1.5  # GB per additional material
        
        estimated_usage = base_usage + (len(active_settings) - 1) * per_material_usage
        return estimated_usage

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False, extra_shading_token=None):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input,
            extra_shading_token=extra_shading_token
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False, extra_shading_token=None):
        self.seed_everything(0)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        
        # Get active PBR settings for output processing
        active_pbr_settings = self.get_active_pbr_settings()
        print(f"Multiview generation using PBR settings: {active_pbr_settings}")
        
        if not isinstance(input_images, List):
            input_images = [input_images]
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None

        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 30,
            "UniPCMultistepScheduler": 15,
            "DDIMScheduler": 50,
            "ShiftSNRScheduler": 15,
        }
        # Move extra shading token to device
        extra_shading_token = extra_shading_token.to(self.pipeline.device) if extra_shading_token is not None else None
        if extra_shading_token is not None:
            extra_shading_token = extra_shading_token.half()

        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=infer_steps_dict[self.pipeline.scheduler.__class__.__name__],
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=3.0,
            extra_shading_token=extra_shading_token,
            **kwargs,
        ).images

        if "pbr" in self.mode:
            # Dynamically organize output based on active PBR settings
            mvd_image_dict = {}
            
            # Calculate expected output count per material
            expected_total = num_view * len(active_pbr_settings)
            if len(mvd_image) != expected_total:
                print(f"Warning: Expected {expected_total} images but got {len(mvd_image)}")
                # Fallback to original behavior if mismatch
                mvd_image_dict = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            else:
                # Organize images by active materials
                for i, material in enumerate(active_pbr_settings):
                    start_idx = i * num_view
                    end_idx = (i + 1) * num_view
                    mvd_image_dict[material] = mvd_image[start_idx:end_idx]
            
            mvd_image = mvd_image_dict
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image
