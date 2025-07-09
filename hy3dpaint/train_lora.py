#!/usr/bin/env python3
"""
LORA fine-tuning for Hunyuan3D-Paint model.

This script uses PEFT (Parameter Efficient Fine-Tuning) to train a LORA adapter
for the UNet model, allowing efficient fine-tuning on custom datasets.

Usage:
    python train_lora.py --config config.yaml --dataset_json path/to/dataset.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import huggingface_hub
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline, DDPMScheduler
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import shutil
import gc

SD_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "proj",
    "proj_in",
    "proj_out",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "to_out.0",
    "time_emb_proj",
    "ff.net.2",
]

# Optional imports with graceful fallback
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT not available. Install with: pip install peft")
    PEFT_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: Weights & Biases not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False

# Local imports
from src.data.dataloader.objaverse_loader_forTexturePBR import TextureDataset
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from utils.torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

def extract_into_tensor(a, t, x_shape):
    """Extract values from tensor a at indices t, reshape to match x_shape"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class TrainingWrapper:
    """Simple wrapper around the pipeline to provide training methods compatible with model.py"""
    
    def __init__(self, pipeline, mr_loss_enabled=False, consistency_loss_lambda=0.15, 
                 albedo_loss_lambda=0.85, view_size=512, num_view=6, device="cuda"):
        self.pipeline = pipeline
        self.unet = pipeline.unet
        self.mr_loss_enabled = mr_loss_enabled
        self.consistency_loss_lambda = consistency_loss_lambda
        self.albedo_loss_lambda = albedo_loss_lambda
        self.view_size = view_size
        self.num_view = num_view
        self.device = device
        self.pbr_settings = pipeline.pbr_settings
        
        # Initialize dino_v2 as None, will be set later if needed
        self.dino_v2 = None
        
        # For compatibility with model.py methods
        self.global_step = 0
        self._mock_optimizer = None
        
        # Set up training scheduler
        from diffusers import DDPMScheduler
        self.train_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        self.register_schedule()
        
    def register_schedule(self):
        """Register noise schedule buffers (from model.py)"""
        self.num_timesteps = self.train_scheduler.config.num_train_timesteps
        betas = self.train_scheduler.betas.detach().cpu()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).float())
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float())
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1).float())
    
    def register_buffer(self, name, tensor):
        """Simple buffer registration"""
        setattr(self, name, tensor.to(self.device))
        
    def to(self, device):
        """Move model to device"""
        self.device = device
        self.pipeline.to(device)
        return self
        
    def set_optimizer(self, optimizer):
        """Set optimizer reference for compatibility"""
        self._mock_optimizer = optimizer
        
    def prepare_batch_data(self, batch):
        """Simplified version of prepare_batch_data from model.py"""
        images_cond = batch["images_cond"].to(self.device)
        cond_imgs, cond_imgs_another = images_cond[:, 0:1, ...], images_cond[:, 1:2, ...]
        
        # Resize to view_size
        from torchvision.transforms import v2
        cond_imgs = v2.functional.resize(cond_imgs, self.view_size, interpolation=3, antialias=True).clamp(0, 1)
        cond_imgs_another = v2.functional.resize(cond_imgs_another, self.view_size, interpolation=3, antialias=True).clamp(0, 1)
        
        target_imgs = {}
        # Always ensure both albedo and mr are available (even if mr is dummy)
        for pbr_token in ["albedo", "mr"]:
            if f"images_{pbr_token}" in batch:
                target_imgs[pbr_token] = batch[f"images_{pbr_token}"].to(self.device)
                target_imgs[pbr_token] = v2.functional.resize(
                    target_imgs[pbr_token], self.view_size, interpolation=3, antialias=True
                ).clamp(0, 1)
            elif pbr_token == "mr" and not self.mr_loss_enabled:
                # Create dummy MR data if not available and not training MR
                if "albedo" in target_imgs:
                    target_imgs[pbr_token] = torch.zeros_like(target_imgs["albedo"])
                else:
                    # Fallback: create dummy based on conditioning images shape
                    B, N_ref = cond_imgs.shape[:2]
                    C, H, W = cond_imgs.shape[2:]
                    target_imgs[pbr_token] = torch.zeros((B, self.num_view, C, H, W), device=self.device)
        
        images_normal = None
        if "images_normal" in batch:
            images_normal = batch["images_normal"].to(self.device)
            images_normal = v2.functional.resize(images_normal, self.view_size, interpolation=3, antialias=True).clamp(0, 1)
            images_normal = [images_normal]
            
        images_position = None
        if "images_position" in batch:
            images_position = batch["images_position"].to(self.device)
            images_position = v2.functional.resize(images_position, self.view_size, interpolation=3, antialias=True).clamp(0, 1)
            images_position = [images_position]
            
        return cond_imgs, cond_imgs_another, target_imgs, images_normal, images_position
        
    def training_step(self, batch, batch_idx):
        """Simplified training step that focuses on albedo loss but provides dummy MR data"""
        # Use the simplified training approach focused on albedo
        cond_imgs, cond_imgs_another, target_imgs, normal_imgs, position_imgs = self.prepare_batch_data(batch)
        
        B = cond_imgs.shape[0]
        
        # Get albedo data
        albedo_imgs = target_imgs["albedo"]  # (B, N, C, H, W)
        mr_imgs = target_imgs["mr"]  # (B, N, C, H, W) - real or dummy data
        
        # Encode images to latents
        albedo_latents = self.encode_images(albedo_imgs)  # (B, N, C, H, W)
        mr_latents = self.encode_images(mr_imgs)  # (B, N, C, H, W) - dummy if no MR data
        ref_latents = self.encode_images(cond_imgs)
        ref_latents_another = self.encode_images(cond_imgs_another)
        
        # Sample timesteps and add noise
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        
        # Stack albedo and MR latents for the expected format: (B, N_pbr, N_gen, C, H, W)
        N_pbr = 2  # Both albedo and MR (even if MR is dummy)
        N_gen = self.num_view  # Number of views
        C, H, W = albedo_latents.shape[2:]
        
        # Stack albedo and MR latents: (B, 2, N, C, H, W)
        gen_latents = torch.stack([albedo_latents, mr_latents], dim=1)  # (B, N_pbr, N_gen, C, H, W)
        
        # Create noise with the same shape
        noise = torch.randn_like(gen_latents)
        
        # Add noise - flatten for noise scheduler, then reshape back
        gen_flat = gen_latents.view(-1, C, H, W)
        noise_flat = noise.view(-1, C, H, W)
        t_flat = t.repeat_interleave(N_pbr * N_gen)  # Expand timesteps for all views and PBR channels
        
        # Add noise
        noisy_latents_flat = self.train_scheduler.add_noise(gen_flat, noise_flat, t_flat)
        
        # Reshape back to expected UNet format
        noisy_latents = noisy_latents_flat.view(B, N_pbr, N_gen, C, H, W)
        
        # Prepare cached condition dict
        cached_condition = {}
        
        # Add reference latents
        cached_condition["ref_latents"] = ref_latents
        
        # Compute DINO features if required
        if hasattr(self.unet, 'use_dino') and self.unet.use_dino and self.dino_v2 is not None:
            # Use first view for DINO features: shape (B, 1, C, H, W)
            dino_hidden_states = self.dino_v2(cond_imgs[:, :1, ...])
            cached_condition["dino_hidden_states"] = dino_hidden_states
        elif hasattr(self.unet, 'use_dino') and self.unet.use_dino:
            print("Warning: Model requires DINO but dino_v2 is not available")
        
        # Prepare text embeddings
        if hasattr(self.unet, 'learned_text_clip_albedo'):
            text_embeds = self.unet.learned_text_clip_albedo.unsqueeze(0).repeat(B, 1, 1)
        else:
            # Fallback to text encoding
            text_embeds = self.pipeline.encode_prompt(["high quality"] * B, self.device, 1, False)[0]
        
        # Add shading embeddings for both albedo and MR (matching model.py)
        if hasattr(self.unet, 'learned_text_clip_albedo'):
            # Build shading embeddings like in the original model
            all_shading_tokens = []
            for token in ["albedo", "mr"]:  # Both tokens even if only training albedo
                all_shading_tokens.append(
                    getattr(self.unet, f"learned_text_clip_{token}").unsqueeze(dim=0).repeat(B, 1, 1)
                )
            cached_condition["shading_embeds"] = torch.stack(all_shading_tokens, dim=1)
        else:
            # Use text embeddings as shading embeds fallback
            # Stack for both albedo and MR
            cached_condition["shading_embeds"] = torch.stack([text_embeds, text_embeds], dim=1)
        
        # Add required attention scaling parameters
        cached_condition["mva_scale"] = 1.0
        cached_condition["ref_scale"] = 1.0
        
        # Add number of views for multiview attention
        cached_condition["num_in_batch"] = N_gen
        
        # Handle normal and position maps if available
        if normal_imgs is not None:
            normal_embeds = self.encode_images(normal_imgs[0])
            cached_condition["embeds_normal"] = normal_embeds
        
        if position_imgs is not None:
            position_embeds = self.encode_images(position_imgs[0])
            cached_condition["embeds_position"] = position_embeds
            cached_condition["position_maps"] = position_imgs[0]
        
        # UNet forward pass with correct input format and cached condition
        # Extract shading_embeds before passing cached_condition to avoid conflicts
        shading_embeds = cached_condition.pop("shading_embeds")
        
        # The custom UNet expects 6D input: (B, N_pbr, N_gen, C, H, W)
        # The shading_embeds should be in format: (B, N_pbr, seq_len, embed_dim)
        # Keep the 4D format since we have both albedo and MR tokens
        
        noise_pred = self.pipeline.unet(
            noisy_latents,  # Pass the 6D tensor: (B, N_pbr, N_gen, C, H, W)
            t_flat,              # Pass the original timesteps: (B,) - UNet handles internal expansion
            encoder_hidden_states=shading_embeds,
            **cached_condition  # Pass all cached condition as kwargs like in forward_unet
        )
        
        # Handle different return formats - could be tuple or object with .sample
        if isinstance(noise_pred, tuple):
            noise_pred = noise_pred[0]
        elif hasattr(noise_pred, 'sample'):
            noise_pred = noise_pred.sample
        
        # Split predictions into albedo and MR components
        noise_pred_flat = noise_pred  # (B*N_pbr*N_gen, C, H, W)
        noise_pred_reshaped = noise_pred_flat.view(B, N_pbr, N_gen, C, H, W)
        noise_pred_albedo = noise_pred_reshaped[:, 0, :, :, :, :]  # (B, N_gen, C, H, W)
        noise_pred_mr = noise_pred_reshaped[:, 1, :, :, :, :]  # (B, N_gen, C, H, W)
        
        # Split targets into albedo and MR components  
        noise_target_reshaped = noise.view(B, N_pbr, N_gen, C, H, W)
        noise_target_albedo = noise_target_reshaped[:, 0, :, :, :, :]  # (B, N_gen, C, H, W)
        noise_target_mr = noise_target_reshaped[:, 1, :, :, :, :]  # (B, N_gen, C, H, W)
        
        # Proper v-prediction loss computation
        if self.train_scheduler.config.prediction_type == "v_prediction":
            # Compute v-targets properly like in model.py
            gen_flat = gen_latents.view(-1, C, H, W)
            v_target = self.get_v(gen_flat, noise_flat, t_flat)
            v_target_reshaped = v_target.view(B, N_pbr, N_gen, C, H, W)
            v_target_albedo = v_target_reshaped[:, 0, :, :, :, :]
            v_target_mr = v_target_reshaped[:, 1, :, :, :, :]
            
            # Compute losses for both albedo and MR (even if only training albedo)
            albedo_loss = F.mse_loss(noise_pred_albedo.view(-1, C, H, W), v_target_albedo.view(-1, C, H, W))
            
            if self.mr_loss_enabled:
                mr_loss = F.mse_loss(noise_pred_mr.view(-1, C, H, W), v_target_mr.view(-1, C, H, W))
            else:
                # Use dummy MR loss if not training MR
                mr_loss = torch.tensor(0.0, device=self.device)
            
            # Apply the same loss weighting as model.py
            loss = self.albedo_loss_lambda * albedo_loss + (1.0 - self.albedo_loss_lambda) * mr_loss
            
        else:
            # Epsilon prediction (standard noise prediction)
            albedo_loss = F.mse_loss(noise_pred_albedo.view(-1, C, H, W), noise_target_albedo.view(-1, C, H, W))
            
            if self.mr_loss_enabled:
                mr_loss = F.mse_loss(noise_pred_mr.view(-1, C, H, W), noise_target_mr.view(-1, C, H, W))
            else:
                mr_loss = torch.tensor(0.0, device=self.device)
            
            loss = self.albedo_loss_lambda * albedo_loss + (1.0 - self.albedo_loss_lambda) * mr_loss
        
        return loss
        
    def get_v(self, x, noise, t):
        """Compute the target velocity (v) for v-prediction training - copied from model.py"""
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
        
    def encode_images(self, images):
        """Encode images to latent space"""
        with torch.no_grad():
            B, N = images.shape[:2]
            images_flat = images.view(-1, *images.shape[2:])
            images_flat = (images_flat - 0.5) * 2.0  # Normalize to [-1, 1]
            
            latents = self.pipeline.vae.encode(images_flat.to(self.pipeline.vae.dtype)).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor
            latents = latents.view(B, N, *latents.shape[1:])
            
        return latents


class LoraTrainer:
    """LORA trainer for Hunyuan3D-Paint model using the model's training_step."""
    
    def __init__(self, 
                 dataset_json: str,
                 output_dir: str = "lora_outputs",
                 batch_size: int = 1,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 3,
                 save_every: int = 1,
                 mixed_precision: bool = True,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 lora_rank: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 use_wandb: bool = False,
                 wandb_project: str = "hunyuan3d-lora",
                 resume_from: str = None,
                 mr_loss_enabled: bool = False,
                 consistency_loss_lambda: float = 0.15,
                 albedo_loss_lambda: float = 0.85,
                 enable_xformers: bool = True,
                 enable_sliced_vae: bool = True,
                 cpu_offload: bool = False):
        """
        Initialize the LORA trainer.
        
        Args:
            dataset_json: Path to dataset JSON file
            output_dir: Directory to save outputs
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            mixed_precision: Use mixed precision training
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            lora_rank: LORA rank
            lora_alpha: LORA alpha parameter
            lora_dropout: LORA dropout rate
            use_wandb: Use Weights & Biases logging
            wandb_project: W&B project name
            resume_from: Path to checkpoint to resume from
            mr_loss_enabled: Whether to enable metallic-roughness loss
            consistency_loss_lambda: Weight for consistency loss
            albedo_loss_lambda: Weight for albedo loss
            enable_xformers: Enable xFormers memory efficient attention
            enable_sliced_vae: Enable sliced VAE decoding for memory efficiency
            cpu_offload: Enable CPU offloading for non-trainable components
        """
        self.dataset_json = dataset_json
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.resume_from = resume_from
        self.mr_loss_enabled = mr_loss_enabled
        self.consistency_loss_lambda = consistency_loss_lambda
        self.albedo_loss_lambda = albedo_loss_lambda
        self.enable_xformers = enable_xformers
        self.enable_sliced_vae = enable_sliced_vae
        self.cpu_offload = cpu_offload
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def setup_model(self):
        """Setup the HunyuanPaint model using the same approach as demo.py."""
        print("Setting up HunyuanPaint model...")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Initialize using the same approach as demo.py
        max_num_view = 6
        resolution = 512
        
        conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        
        # Load the pretrained model the same way multiview_utils.py does
        model_path = huggingface_hub.snapshot_download(
            repo_id=conf.multiview_pretrained_path,
            allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
        )
        model_path = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")
        
        # Load the pipeline with pretrained weights
        # Use CPU first to avoid OOM during loading, then move to GPU
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=conf.custom_pipeline,
            torch_dtype=torch.float16,
            device_map=None  # Load to CPU first
        )
        
        # Set up the scheduler
        from diffusers import UniPCMultistepScheduler
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )
        pipeline.set_progress_bar_config(disable=True)
        
        # Enable gradient checkpointing to reduce memory usage
        if hasattr(pipeline.unet, 'enable_gradient_checkpointing'):
            pipeline.unet.enable_gradient_checkpointing()
            print("Enabled gradient checkpointing for UNet")
        
        if hasattr(pipeline.vae, 'enable_gradient_checkpointing'):
            pipeline.vae.enable_gradient_checkpointing()
            print("Enabled gradient checkpointing for VAE")
        
        # Enable memory optimizations
        if self.enable_xformers:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("Enabled xFormers memory efficient attention")
            except Exception as e:
                print(f"Failed to enable xFormers: {e}")
        
        if self.enable_sliced_vae:
            try:
                pipeline.vae.enable_slicing()
                print("Enabled sliced VAE decoding")
            except Exception as e:
                print(f"Failed to enable sliced VAE: {e}")
        
        # Enable CPU offloading for non-trainable components
        if self.cpu_offload:
            try:
                pipeline.enable_model_cpu_offload()
                print("Enabled CPU offloading for non-trainable components")
            except Exception as e:
                print(f"Failed to enable CPU offloading: {e}")
        
        # Set PBR settings - always use both albedo and MR for model compatibility
        # Even if we only train on albedo, the model expects both channels
        pipeline.set_pbr_settings(["albedo", "mr"])
        
        # Create our training wrapper
        self.model = TrainingWrapper(
            pipeline=pipeline,
            mr_loss_enabled=self.mr_loss_enabled,
            consistency_loss_lambda=self.consistency_loss_lambda,
            albedo_loss_lambda=self.albedo_loss_lambda,
            view_size=resolution,
            num_view=max_num_view,
            device=self.device
        )
        
        # Load DINO if available
        if hasattr(pipeline.unet, "use_dino") and pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            self.model.dino_v2 = Dino_v2(conf.dino_ckpt_path).to(torch.float16)
            self.model.dino_v2 = self.model.dino_v2.to(self.device)
            print(f"DINO v2 model loaded successfully from {conf.dino_ckpt_path}")
        else:
            print("DINO v2 not required by this model")
        
        # Move to device after all setup is complete
        pipeline = pipeline.to(self.device)
        
        # Clear cache after moving to device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Setup mixed precision
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"HunyuanPaint model loaded successfully with pretrained weights")
        print(f"Model path: {model_path}")
        print(f"PBR settings: {self.model.pbr_settings}")
        print(f"MR loss enabled: {self.model.mr_loss_enabled}")
        
    def setup_lora(self):
        """Setup LORA adapter for the UNet."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required for LORA training. Install with: pip install peft")
            
        print("Setting up LORA adapter...")
        
        # Define LORA config
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=SD_TARGET_MODULES,
            lora_dropout=self.lora_dropout,
            bias="none",
        )
        
        # Apply LORA to UNet
        self.model.unet = get_peft_model(self.model.unet, lora_config)
        
        # Enable training mode for LORA parameters only
        self.model.unet.train()

        # Freeze other components
        self.model.pipeline.vae.eval()
        self.model.pipeline.text_encoder.eval()
        for param in self.model.pipeline.vae.parameters():
            param.requires_grad = False
        for param in self.model.pipeline.text_encoder.parameters():
            param.requires_grad = False
            
        # Freeze DINO if present
        if hasattr(self.model, 'dino_v2'):
            self.model.dino_v2.eval()
            for param in self.model.dino_v2.parameters():
                param.requires_grad = False
            
        print(f"LORA adapter applied to UNet")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.unet.parameters() if p.requires_grad):,}")
        
    def setup_dataset(self):
        """Setup dataset and dataloader."""
        print("Setting up dataset...")
        
        # Create dataset
        self.dataset = TextureDataset(
            json_path=self.dataset_json,
            num_view=6,
            image_size=512,
            lighting_suffix_pool=["light_PL", "light_AL", "light_ENVMAP", "light_KEY"]
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        print(f"Dataset created with {len(self.dataset)} samples")
        print(f"DataLoader created with batch size {self.batch_size}")
        
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        print("Setting up optimizer...")
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.unet.parameters() if p.requires_grad]
        
        # If using learned text clip, also include those parameters in training
        if hasattr(self.model.unet, 'use_learned_text_clip') and self.model.unet.use_learned_text_clip:
            for param_name in dir(self.model.unet):
                if param_name.startswith('learned_text_clip_'):
                    param = getattr(self.model.unet, param_name)
                    if isinstance(param, torch.nn.Parameter):
                        param.requires_grad = True
                        trainable_params.append(param)
                        print(f"Added {param_name} to trainable parameters")
        
        # Create optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )
        
        # Create learning rate scheduler
        total_steps = len(self.dataloader) * self.num_epochs // self.gradient_accumulation_steps
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.learning_rate * 0.1
        )
        
        print(f"Optimizer created with {len(trainable_params)} parameter groups")
        print(f"Total training steps: {total_steps}")
        
    def setup_logging(self):
        """Setup logging with Weights & Biases."""
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: Weights & Biases not available. Disabling wandb logging.")
                self.use_wandb = False
                return
                
            wandb.init(
                project=self.wandb_project,
                config={
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "num_epochs": self.num_epochs,
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "lora_dropout": self.lora_dropout,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "mixed_precision": self.mixed_precision
                }
            )
            
    def prepare_batch_data(self, batch):
        """Use the model's prepare_batch_data method."""
        return self.model.prepare_batch_data(batch)
    
    def training_step(self, batch):
        """Perform a single training step using the model's training_step method."""
        # Move batch to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
        
        # Set the model's optimizer reference for compatibility
        self.model.set_optimizer(self.optimizer)
        
        # Forward pass using the model's training_step
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                loss = self.model.training_step(batch, 0)  # batch_idx=0
                loss = loss / self.gradient_accumulation_steps
        else:
            loss = self.model.training_step(batch, 0)  # batch_idx=0
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Extract loss components for logging (if available from the model)
        loss_dict = {
            "loss_total": loss.item() * self.gradient_accumulation_steps
        }
        
        return loss_dict
    
    def save_checkpoint(self, epoch):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save LORA adapter
        self.model.unet.save_pretrained(checkpoint_dir)
        
        # Save training state
        state_dict = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        
        if self.mixed_precision:
            state_dict["scaler"] = self.scaler.state_dict()
            
        torch.save(state_dict, checkpoint_dir / "training_state.pt")
        
        # Save config
        config_dict = {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "mr_loss_enabled": self.mr_loss_enabled,
            "consistency_loss_lambda": self.consistency_loss_lambda,
            "albedo_loss_lambda": self.albedo_loss_lambda
        }
        
        with open(checkpoint_dir / "training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint with proper memory management."""
        checkpoint_path = Path(checkpoint_path)
        
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load training state first (lighter memory footprint)
        state_file = checkpoint_path / "training_state.pt"
        if state_file.exists():
            print("Loading training state...")
            # Load to CPU first to avoid GPU memory spike
            state_dict = torch.load(state_file, map_location='cpu')
            self.epoch = state_dict["epoch"]
            self.global_step = state_dict["global_step"]
            
            # Load optimizer state (this can be memory intensive)
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                self.optimizer.load_state_dict(state_dict["optimizer"])
            
            # Load scheduler state
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            
            # Load scaler state if using mixed precision
            if self.mixed_precision and "scaler" in state_dict and hasattr(self, 'scaler'):
                self.scaler.load_state_dict(state_dict["scaler"])
            
            # Clear the state_dict from memory
            del state_dict
        
        # Clear cache before loading LORA weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load LORA adapter with proper memory management
        print("Loading LORA adapter...")
        
        # Check if LORA config file exists
        if (checkpoint_path / "adapter_config.json").exists():
            try:
                # Load adapter with explicit adapter name to avoid conflicts
                adapter_name = "resumed_adapter"
                
                # If adapter already exists, remove it first
                if hasattr(self.model.unet, 'peft_config') and adapter_name in self.model.unet.peft_config:
                    print(f"Removing existing adapter: {adapter_name}")
                    self.model.unet.delete_adapter(adapter_name)
                
                # Load the new adapter
                self.model.unet.load_adapter(checkpoint_path, adapter_name, device=self.device)
                self.model.unet.set_adapter(adapter_name)
                
                print(f"LORA adapter loaded and activated: {adapter_name}")
                
            except Exception as e:
                print(f"Error loading LORA adapter: {e}")
                print("Attempting alternative loading method...")
                
                # Alternative method: load weights manually
                try:
                    adapter_weights = torch.load(checkpoint_path / "adapter_model.safetensors", map_location=self.device)
                    # Load weights into the current LORA adapter
                    missing_keys, unexpected_keys = self.model.unet.load_state_dict(adapter_weights, strict=False)
                    if missing_keys:
                        print(f"Missing keys during LORA loading: {missing_keys[:5]}...")  # Show first 5
                    if unexpected_keys:
                        print(f"Unexpected keys during LORA loading: {unexpected_keys[:5]}...")  # Show first 5
                except Exception as e2:
                    print(f"Alternative loading method also failed: {e2}")
                    print("Continuing without loaded LORA weights...")
        
        # Ensure model is in training mode and on correct device
        self.model.unet.train()
        self.model.to(self.device)
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
                
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, global step {self.global_step}")
        
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        # Setup all components
        self.setup_model()
        self.setup_lora()
        self.setup_dataset()
        self.setup_optimizer()
        self.setup_logging()
        
        # Clear cache after setup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Resume from checkpoint if specified
        if self.resume_from:
            self.load_checkpoint(self.resume_from)
            
            # Additional cleanup after checkpoint loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Training loop
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                loss_dict = self.training_step(batch)
                epoch_loss += loss_dict["loss_total"]
                num_batches += 1
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.unet.parameters() if p.requires_grad], 
                        self.max_grad_norm
                    )
                    
                    # Optimizer step
                    if self.mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Clear cache periodically to prevent memory buildup
                    if self.global_step % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                        print(f"Step {self.global_step}: out of step {len(self.dataloader)}")

                # Update progress bar
                avg_loss = epoch_loss / num_batches
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Log to wandb
                if self.use_wandb and WANDB_AVAILABLE and (step + 1) % self.gradient_accumulation_steps == 0:
                    wandb.log({
                        "train/loss": loss_dict["loss_total"],
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": self.global_step
                    })
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        # Save final checkpoint
        self.save_checkpoint(self.num_epochs)
        print("Training completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LORA fine-tuning for Hunyuan3D-Paint")
    
    # Required arguments
    parser.add_argument("--dataset_json", type=str, required=True,
                       help="Path to dataset JSON file")
    
    # Optional arguments
    parser.add_argument("--config", type=str, default=None,
                       help="Path to model config YAML file")
    parser.add_argument("--output_dir", type=str, default="lora_outputs",
                       help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for training (use 1 for memory constrained setups)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--save_every", type=int, default=1,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps (increase if using smaller batch size)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LORA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LORA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LORA dropout rate")
    parser.add_argument("--no_mixed_precision", action="store_true",
                       help="Disable mixed precision training")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="hunyuan3d-lora",
                       help="Weights & Biases project name")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--mr_loss_enabled", action="store_true",
                       help="Enable metallic-roughness loss (default: only albedo)")
    parser.add_argument("--consistency_loss_lambda", type=float, default=0.15,
                       help="Weight for consistency loss")
    parser.add_argument("--albedo_loss_lambda", type=float, default=0.85,
                       help="Weight for albedo loss")
    parser.add_argument("--enable_xformers", action="store_true", default=False,
                       help="Enable xFormers memory efficient attention")
    parser.add_argument("--disable_xformers", action="store_true",
                       help="Disable xFormers memory efficient attention")
    parser.add_argument("--enable_sliced_vae", action="store_true", default=True,
                       help="Enable sliced VAE decoding for memory efficiency")
    parser.add_argument("--disable_sliced_vae", action="store_true",
                       help="Disable sliced VAE decoding")
    parser.add_argument("--cpu_offload", action="store_true",
                       help="Enable CPU offloading for non-trainable components (saves GPU memory)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.dataset_json):
        print(f"Error: Dataset JSON file not found: {args.dataset_json}")
        return 1
    
    # Create trainer
    trainer = LoraTrainer(
        dataset_json=args.dataset_json,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        mixed_precision=not args.no_mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        resume_from=args.resume_from,
        mr_loss_enabled=args.mr_loss_enabled,
        consistency_loss_lambda=args.consistency_loss_lambda,
        albedo_loss_lambda=args.albedo_loss_lambda,
        enable_xformers=args.enable_xformers and not args.disable_xformers,
        enable_sliced_vae=args.enable_sliced_vae and not args.disable_sliced_vae,
        cpu_offload=args.cpu_offload
    )
    
    # Start training
    try:
        trainer.train()
        print("Training completed successfully!")
        return 0
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
