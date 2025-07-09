#!/usr/bin/env python3
"""
Inference script for using trained LORA adapters with Hunyuan3D-Paint.

This script demonstrates how to load and use a trained LORA adapter for texture generation.
"""

import os
import sys
import argparse
import torch
from PIL import Image
from pathlib import Path

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
        
        # Load LORA adapter into the UNet
        unet = self.paint_pipeline.models["multiview_model"].pipeline.unet
        self.lora_unet = PeftModel.from_pretrained(unet, self.lora_checkpoint_path)
        
        # Replace the UNet in the pipeline
        self.paint_pipeline.models["multiview_model"].pipeline.unet = self.lora_unet
        
        print("LORA adapter loaded successfully")
        
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
        
        # Generate texture using the pipeline
        result_path = self.paint_pipeline(
            mesh_path=mesh_path,
            image_path=image_path,
            output_mesh_path=output_mesh_path,
            use_remesh=True,
            save_glb=True
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
            output_mesh_path=args.output_mesh
        )
        
        print(f"\n{'='*50}")
        print("Texture generation completed successfully!")
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
