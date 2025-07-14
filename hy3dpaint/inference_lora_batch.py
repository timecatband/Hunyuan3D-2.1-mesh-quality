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

# Local imports
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
import gc

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
        
    def generate_texture(self, mesh_path: str, image_path: str, output_mesh_path: str = None):
        """
        Generate texture for a mesh using the LORA-adapted model.
        
        Args:
            mesh_path: Path to input mesh
            image_path: Path to reference image
            output_mesh_path: Path for output mesh (optional)
            
        Returns:
            Path to the generated textured mesh
        """
        print(f"Generating texture for mesh: {mesh_path}")
        print(f"Using reference image: {image_path}")
        
        # Load and display training config
        training_config = self.get_training_config()
        
        # Verify model state before generation
        self.verify_model_state()
        
        # Generate texture using the pipeline
        result_path = self.paint_pipeline(
            mesh_path=mesh_path,
            image_path=image_path,
            output_mesh_path=output_mesh_path,
            use_remesh=True,
            save_glb=True,
            learnable_shading_token=self.learnable_shading_token
        )
        
        print(f"Textured mesh saved to: {result_path}")
        return result_path
        
    def batch_generate(self, mesh_list: list, image_list: list, output_dir: str):
        """
        Generate textures for multiple meshes.
        
        Args:
            mesh_list: List of mesh file paths
            image_list: List of reference image paths
            output_dir: Output directory for generated meshes
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (mesh_path, image_path) in enumerate(zip(mesh_list, image_list)):
            print(f"\nProcessing {i+1}/{len(mesh_list)}: {mesh_path}")
            
            # Generate output filename
            mesh_name = Path(mesh_path).stem
            output_mesh_path = output_dir / f"{mesh_name}_textured.obj"
            
            try:
                result_path = self.generate_texture(
                    mesh_path=mesh_path,
                    image_path=image_path, 
                    output_mesh_path=str(output_mesh_path)
                )
                print(f"✓ Success: {result_path}")
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"✗ Failed: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="LORA batch inference for Hunyuan3D-Paint")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to trained LORA checkpoint directory")
    parser.add_argument("--input_dir",       type=str, required=True, help="Root directory with subdirs, each containing a .glb and an image")
    parser.add_argument("--output_mesh", type=str, default=None, help="(unused)")
    parser.add_argument("--max_num_view", type=int, default=6, help="Maximum number of views (default: 6)")
    parser.add_argument("--resolution",   type=int, default=512, help="Output resolution (default: 512)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.input_dir):
        print(f"Error: input_dir not found: {args.input_dir}")
        return 1
    if not os.path.exists(args.lora_checkpoint):
        print(f"Error: LORA checkpoint not found: {args.lora_checkpoint}")
        return 1

    try:
        # Create inference pipeline
        pipeline = LoraInferencePipeline(
            lora_checkpoint_path=args.lora_checkpoint,
            max_num_view=args.max_num_view,
            resolution=args.resolution
        )

        # Iterate each subdirectory
        for sub in sorted(os.listdir(args.input_dir)):
            subp = os.path.join(args.input_dir, sub)
            if not os.path.isdir(subp): continue
            # find .glb
            glb_list = [f for f in os.listdir(subp) if f.lower().endswith(".glb")]
            if not glb_list:
                print(f"No .glb in {subp}, skipping")
                continue
            mesh_path = os.path.join(subp, glb_list[0])
            # find image
            img_list = [f for f in os.listdir(subp) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not img_list:
                print(f"No image in {subp}, skipping")
                continue
            image_path = os.path.join(subp, img_list[0])
            out_path = os.path.join(subp, Path(glb_list[0]).stem + "_textured.obj")
            print(f"\nProcessing {subp}")
            try:
                pipeline.generate_texture(mesh_path, image_path, output_mesh_path=out_path)
                print(f"✓ Saved textured mesh to {out_path}")
                # free GPU memory between runs
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"✗ Failed for {subp}: {e}")
        return 0
    except Exception as e:
        print(f"Error: batch inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
   sys.exit(main())

