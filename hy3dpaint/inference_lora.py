#!/usr/bin/env python3
"""
Inference script for using trained LORA adapters with Hunyuan3D-Paint.

This script demonstrates how to load and use a trained LORA adapter for texture generation.
"""

import os
import sys
import argparse
import torch
import json
from PIL import Image
from pathlib import Path
from typing import List

# Local imports
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT not available. Install with: pip install peft")
    PEFT_AVAILABLE = False

try:
    from utils.torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


class LoraInferencePipeline:
    """Pipeline for inference with LORA adapters."""
    
    def __init__(self, lora_checkpoint_path: str, max_num_view: int = 6, resolution: int = 512):
        """
        Initialize the LORA inference pipeline.
        
        Args:
            lora_checkpoint_path: Path to the trained LORA checkpoint
            max_num_view: Maximum number of views to generate
            resolution: Output resolution
        """
        self.lora_checkpoint_path = Path(lora_checkpoint_path)
        self.max_num_view = max_num_view
        self.resolution = resolution
        
        # Setup base pipeline
        self.setup_base_pipeline()
        
        # Load LORA adapter
        self.load_lora_adapter()
        self.load_shading_token()

    def setup_base_pipeline(self):
        """Setup the base Hunyuan3D-Paint pipeline."""
        print("Setting up base pipeline...")
        
        # Create config
        conf = Hunyuan3DPaintConfig(self.max_num_view, self.resolution)
        
        # Create pipeline
        self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
        
        print("Base pipeline loaded successfully")
        
    def load_lora_adapter(self):
        """Load the trained LORA adapter."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required for LORA inference. Install with: pip install peft")
            
        print(f"Loading LORA adapter from: {self.lora_checkpoint_path}")
        
        # --- Main Generator UNet ---
        # Get the inner UNet, which is the actual target for LoRA modifications
        inner_unet_main = self.paint_pipeline.models["multiview_model"].pipeline.unet
        
        # Load the LoRA weights into the inner UNet using PeftModel.
        lora_unet_main = PeftModel.from_pretrained(inner_unet_main, self.lora_checkpoint_path)
        
        # Merge the LoRA weights into the base model.
        print("Merging LoRA adapter into main generator UNet...")
        self.paint_pipeline.models["multiview_model"].pipeline.unet = lora_unet_main.merge_and_unload()
        
        print("LORA adapter loaded successfully")
        
    def load_shading_token(self):
        """Load the learned albedo shading token if present."""
        token_path = self.lora_checkpoint_path / "learnable_albedo_token.pt"
        if token_path.exists():
            self.learnable_shading_token = torch.load(token_path, map_location="cpu")
            print(f"Loaded shading token from {token_path}")
        else:
            self.learnable_shading_token = None
            print("No shading token found in checkpoint; proceeding without it")

    def get_training_config(self):
        """Load the training configuration if available."""
        config_file = self.lora_checkpoint_path / "training_config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            print("✓ Training configuration loaded:")
            for key, value in config.items():
                print(f"  - {key}: {value}")
            return config
        else:
            print("No training config found")
            return {}
    
    def verify_model_state(self):
        """Verify the model is properly configured with LORA."""
        print("\n=== Model State Verification ===")
        
        unet = self.paint_pipeline.models["multiview_model"].pipeline.unet
        
        # Check LORA status
        if hasattr(unet, 'peft_config'):
            print(f"✓ LORA adapter detected: {list(unet.peft_config.keys())}")
            for adapter_name, config in unet.peft_config.items():
                print(f"  - {adapter_name}: rank={config.r}, alpha={config.lora_alpha}")
        else:
            print("⚠️ No LORA adapter detected")
        
        # Check built-in shading tokens
        if hasattr(unet, 'pbr_setting'):
            print("✓ Using built-in shading tokens:")
            for token in unet.pbr_setting:
                print(f"  - {token}")
        else:
            print("⚠️ No PBR settings found")
        
        print("=================================\n")
        
    def generate_texture(self, mesh_path: str, image_path: str, output_mesh_path: str = None, pbr_settings: List[str] = None):
        """
        Generate texture for a mesh using the LORA-adapted model.
        
        Args:
            mesh_path: Path to input mesh
            image_path: Path to reference image
            output_mesh_path: Path for output mesh (optional)
            pbr_settings: List of PBR materials to generate (e.g., ["albedo"] for albedo-only)
            
        Returns:
            Path to the generated textured mesh
        """
        print(f"Generating texture for mesh: {mesh_path}")
        print(f"Using reference image: {image_path}")
        
        # Configure PBR settings (default to albedo-only for RAM saving)
        if pbr_settings is None:
            pbr_settings = ["albedo"]
            
        # Load and display training config
        training_config = self.get_training_config()
        
        # Verify model state before generation
        self.verify_model_state()
        
        # Configure PBR settings
        self.paint_pipeline.set_active_pbr_settings(pbr_settings)
        
        # Display memory estimate
        estimated_vram = self.paint_pipeline.get_memory_usage_estimate()
        print(f"Estimated VRAM usage: {estimated_vram:.1f} GB")
        
        # Generate texture using the pipeline
        result_path = self.paint_pipeline(
            mesh_path=mesh_path,
            image_path=image_path,
            output_mesh_path=output_mesh_path,
            use_remesh=True,
            save_glb=True,
            learnable_shading_token=self.learnable_shading_token,
            pbr_settings=pbr_settings
        )
        
        print(f"Textured mesh saved to: {result_path}")
        return result_path
        
    def batch_generate(self, mesh_list: list, image_list: list, output_dir: str, pbr_settings: List[str] = None):
        """
        Generate textures for multiple meshes.
        
        Args:
            mesh_list: List of mesh file paths
            image_list: List of reference image paths
            output_dir: Output directory for generated meshes
            pbr_settings: List of PBR materials to generate
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default to albedo-only for batch processing to save memory
        if pbr_settings is None:
            pbr_settings = ["albedo"]
        
        for i, (mesh_path, image_path) in enumerate(zip(mesh_list, image_list)):
            print(f"\nProcessing {i+1}/{len(mesh_list)}: {mesh_path}")
            
            # Generate output filename
            mesh_name = Path(mesh_path).stem
            output_mesh_path = output_dir / f"{mesh_name}_textured.obj"
            
            try:
                result_path = self.generate_texture(
                    mesh_path=mesh_path,
                    image_path=image_path, 
                    output_mesh_path=str(output_mesh_path),
                    pbr_settings=pbr_settings
                )
                print(f"✓ Success: {result_path}")
            except Exception as e:
                print(f"✗ Failed: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="LORA inference for Hunyuan3D-Paint")
    
    # Required arguments
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                       help="Path to trained LORA checkpoint directory")
    parser.add_argument("--mesh_path", type=str, required=True,
                       help="Path to input mesh file")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to reference image")
    
    # Optional arguments
    parser.add_argument("--output_mesh", type=str, default=None,
                       help="Path for output textured mesh")
    parser.add_argument("--max_num_view", type=int, default=6,
                       help="Maximum number of views (default: 6)")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Output resolution (default: 512)")
    parser.add_argument("--pbr_settings", nargs="+", default=["albedo"],
                       choices=["albedo", "mr"],
                       help="PBR materials to generate (default: albedo only)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.lora_checkpoint):
        print(f"Error: LORA checkpoint not found: {args.lora_checkpoint}")
        return 1
        
    if not os.path.exists(args.mesh_path):
        print(f"Error: Mesh file not found: {args.mesh_path}")
        return 1
        
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return 1
    
    try:
        # Create inference pipeline
        pipeline = LoraInferencePipeline(
            lora_checkpoint_path=args.lora_checkpoint,
            max_num_view=args.max_num_view,
            resolution=args.resolution
        )
        
        # Generate texture
        result_path = pipeline.generate_texture(
            mesh_path=args.mesh_path,
            image_path=args.image_path,
            output_mesh_path=args.output_mesh,
            pbr_settings=args.pbr_settings
        )
        
        print(f"\n{'='*50}")
        print("Texture generation completed successfully!")
        print(f"Generated materials: {args.pbr_settings}")
        print(f"Output: {result_path}")
        print(f"{'='*50}")
        
        return 0
        
    except Exception as e:
        print(f"Error: Texture generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
   sys.exit(main())
