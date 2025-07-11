#!/usr/bin/env python3
"""
Data augmentation script for Hunyuan3D-Paint training data.

This script takes a dataset prepared by prepare_data.py and creates augmented copies
by applying light img2img diffusion to the conditioning images in render_cond folders.

Usage:
    python augment_data.py --input_dir path/to/dataset --output_dir path/to/augmented --strength 0.1
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

try:
    from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
    from transformers import BlipProcessor, BlipForConditionalGeneration
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    print(f"Error: diffusers not available: {e}")
    print("Install with: pip install diffusers transformers accelerate")
    DIFFUSERS_AVAILABLE = False


class DataAugmenter:
    """
    Augments Hunyuan3D-Paint training data by applying light img2img diffusion
    to conditioning images while preserving the original structure.
    """
    
    def __init__(self, 
                 model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "cuda",
                 torch_dtype = torch.float16,
                 negative_prompt: str = "",
                 use_auto_caption: bool = False):
        """
        Initialize the data augmenter.
        
        Args:
            model_id: HuggingFace model ID for Stable Diffusion
            device: Device to use for inference
            torch_dtype: Data type for inference
            negative_prompt: Negative prompt for diffusion
        """
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers package is required but not available")
        
        self.device = device
        self.torch_dtype = torch_dtype
        self.negative_prompt = negative_prompt
        self.use_auto_caption = use_auto_caption
        
        print(f"Loading SDXL model: {model_id}")
        
        # Load captioning model if requested
        if self.use_auto_caption:
            print("Loading BLIP captioning model...")
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype
            ).to(self.device)
            print("BLIP captioning model loaded successfully")
        
        # Load ControlNet model
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=self.torch_dtype
        ).to(self.device)

        # Load img2img pipeline with ControlNet
        self.pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
             model_id,
             controlnet=self.controlnet,
             torch_dtype=self.torch_dtype,
         )
         
        self.pipeline = self.pipeline.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention")
            except Exception as e:
                print(f"Could not enable xformers: {e}")
        
        # Enable model offloading to save memory
        if hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()
            print("Enabled model CPU offload")
        
        print("SDXL pipeline loaded successfully")
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate a caption for the image using BLIP."""
        if not self.use_auto_caption:
            return "high quality, detailed texture, rendering, unreal engine"
        
        try:
            inputs = self.caption_processor(image, return_tensors="pt").to(self.device)
            out = self.caption_model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            
            # Enhance the caption for better diffusion results
            enhanced_caption = f"{caption}, high quality, detailed, photorealistic, detailed texture"
            print(f"Generated caption: {enhanced_caption}")
            return enhanced_caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "high quality, detailed texture, photorealistic"

    def augment_image(self, 
                     image_path: Path, 
                     strength: float = 0.1,
                     guidance_scale: float = 7.5,
                     num_inference_steps: int = 20) -> Image.Image:
        """
        Augment a single image using img2img diffusion.
        
        Args:
            image_path: Path to the input image
            strength: Strength of the diffusion (0.0 = no change, 1.0 = complete generation)
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of inference steps
            
        Returns:
            Augmented PIL Image
        """
        # Load image
        original_image = Image.open(image_path).convert("RGB")
        print(f"Original image size: {original_image.size}")
        
        # Ensure dimensions are multiples of 8 for VAE compatibility
        w, h = original_image.size
        new_w = max((w // 8) * 8, 8)
        new_h = max((h // 8) * 8, 8)
        if (new_w, new_h) != (w, h):
            print(f"Resizing from {w}x{h} to {new_w}x{new_h}")
            original_image = original_image.resize((new_w, new_h), Image.LANCZOS)
        
        # Generate Canny edge map for ControlNet via OpenCV
        gray = np.array(original_image.convert("L"))
        edges = cv2.Canny(gray, 100, 200)  # adjust thresholds as needed
        control_image = Image.fromarray(edges)  # Keep as grayscale, then convert
        control_image = control_image.convert("RGB")
        print(f"Control image size: {control_image.size}")

        # Generate prompt using auto-captioning or use default
        prompt = self.generate_caption(original_image)
        print("Num steps:", num_inference_steps)
        num_inference_steps = 50

        # Jitter strength in a 0.1 range around the specified strength
        jitter_strength = strength + np.random.uniform(-0.125, 0.125)

        # Apply img2img diffusion
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                image=original_image,
                control_image=control_image,
                controlnet_conditioning_scale=0.8,  # Reduce control strength
                strength=jitter_strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                # Remove guess_mode - it's experimental and can cause artifacts
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
         
        # Ensure we return the same size as input
        result_image = result.images[0]
        if result_image.size != original_image.size:
            result_image = result_image.resize(original_image.size, Image.LANCZOS)
        return result_image
    
    def copy_example_folder(self, 
                           source_path: Path, 
                           dest_path: Path) -> bool:
        """
        Copy an example folder to create the base for augmentation.
        
        Args:
            source_path: Source example folder
            dest_path: Destination folder path
            
        Returns:
            True if copy succeeded, False otherwise
        """
        try:
            if dest_path.exists():
                shutil.rmtree(dest_path)
            
            shutil.copytree(source_path, dest_path)
            print(f"Copied {source_path} -> {dest_path}")
            return True
            
        except Exception as e:
            print(f"Error copying folder {source_path}: {e}")
            return False
    
    def augment_render_cond_folder(self, 
                                  example_path: Path,
                                  strength: float = 0.1,
                                  num_inference_steps: int = 50) -> bool:
        """
        Augment all images in the render_cond folder of an example.
        
        Args:
            example_path: Path to the example folder
            strength: Diffusion strength for augmentation
            
        Returns:
            True if augmentation succeeded, False otherwise
        """
        render_cond_path = example_path / "render_cond"
        
        if not render_cond_path.exists():
            print(f"Warning: render_cond folder not found in {example_path}")
            return False
        
        # Find all image files in render_cond
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = [
            f for f in render_cond_path.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"Warning: No image files found in {render_cond_path}")
            return False
        
        # Select only first 5 images to process
        images_to_process = image_files[:5]
        images_to_remove = image_files[5:]
        print("Processing all images")
        
        # Remove images we won't process
        for image_file in images_to_remove:
            try:
                image_file.unlink()
                print(f"Removed unprocessed image: {image_file.name}")
            except Exception as e:
                print(f"Error removing {image_file}: {e}")
        
        print(f"Augmenting {len(images_to_process)} images in {render_cond_path}")
        
        success_count = 0
        
        # Process each image
        for image_file in tqdm(images_to_process, desc="Augmenting images"):
            try:
                print(f"Processing image: {image_file}")
                 # Augment the image
                augmented_image = self.augment_image(image_file, strength=strength, num_inference_steps=num_inference_steps)
                
                # Save back to the same location (replacing original)
                augmented_image.save(image_file)
                
                success_count += 1
                
                print(f"✓ Successfully processed: {image_file}")
            except Exception as e:
                print(f"✗ Error processing {image_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
                

        
        print(f"Successfully augmented {success_count}/{len(images_to_process)} images")
        return success_count > 0
    
    def process_dataset(self, 
                       input_dir: str,
                       output_dir: str,
                       strength: float = 0.1,
                       num_inference_steps: int = 50,
                       suffix: str = "_augment") -> bool:
        """
        Process an entire dataset by augmenting all example folders.
        
        Args:
            input_dir: Root directory containing example folders
            output_dir: Directory to save augmented dataset
            strength: Diffusion strength for augmentation
            suffix: Suffix to add to augmented folder names
            
        Returns:
            True if processing succeeded, False otherwise
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            return False
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all example folders (directories containing render_cond and render_tex)
        example_folders = []
        for item in input_path.iterdir():
            if item.is_dir():
                render_cond = item / "render_cond"
                render_tex = item / "render_tex"
                if render_cond.exists() and render_tex.exists():
                    example_folders.append(item)
        
        if not example_folders:
            print(f"Error: No valid example folders found in {input_dir}")
            print("Expected folders with both 'render_cond' and 'render_tex' subdirectories")
            return False
        
        print(f"Found {len(example_folders)} example folders to process")
        
        successful_count = 0
        failed_count = 0
        
        # Process each example folder
        for example_folder in tqdm(example_folders, desc="Processing examples"):
            try:
                example_name = example_folder.name
                augmented_name = f"{example_name}{suffix}"
                
                # Determine output path
                if input_path == output_path:
                    # Save to same directory with suffix
                    dest_path = output_path / augmented_name
                else:
                    # Save to different directory
                    dest_path = output_path / augmented_name
                
                # Skip if already processed
                if dest_path.exists():
                    print(f"Skipping {example_name} -> {augmented_name} (already exists)")
                    successful_count += 1  # Count as successful since it was processed before
                    continue
                
                print(f"\nProcessing: {example_name} -> {augmented_name}")
                
                # Copy the original folder
                if not self.copy_example_folder(example_folder, dest_path):
                    failed_count += 1
                    continue
                
                # Augment the render_cond images
                if self.augment_render_cond_folder(dest_path, strength=strength, num_inference_steps=num_inference_steps):
                    successful_count += 1
                    print(f"✓ Successfully augmented: {augmented_name}")
                else:
                    failed_count += 1
                    print(f"✗ Failed to augment: {augmented_name}")
                
            except Exception as e:
                print(f"Error processing {example_folder}: {e}")
                failed_count += 1
                continue
        
        # Print summary
        print(f"\n{'='*60}")
        print("AUGMENTATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total examples: {len(example_folders)}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {failed_count}")
        
        return successful_count > 0
    
    def cleanup(self):
        """Clean up pipeline resources."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
            torch.cuda.empty_cache()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Augment Hunyuan3D-Paint training data using img2img diffusion")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Root directory containing example folders")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save augmented data (default: same as input_dir)")
    parser.add_argument("--strength", type=float, default=0.1,
                       help="Diffusion strength for augmentation (0.0-1.0, default: 0.1)")
    parser.add_argument("--suffix", type=str, default="_augment",
                       help="Suffix to add to augmented folder names (default: '_augment')")
    parser.add_argument("--model_id", type=str, 
                       default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="HuggingFace model ID for Flux")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference (default: cuda)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale for diffusion (default: 7.5)")
    parser.add_argument("--num_inference_steps", type=int, default=10,
                       help="Number of inference steps (default: 20)")
    parser.add_argument("--negative_prompt", type=str, default="distorted, blurry, low quality, corrupt, ugly, deformed",
                        help="Negative prompt for diffusion (default: none)")
    parser.add_argument("--use_auto_caption", action="store_true",
                        help="Use auto-captioning model to generate prompts")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    if args.strength < 0.0 or args.strength > 1.0:
        print(f"Error: Strength must be between 0.0 and 1.0, got: {args.strength}")
        return 1
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    # Check for CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    try:
        # Create augmenter
        augmenter = DataAugmenter(
            model_id=args.model_id,
            device=args.device,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            negative_prompt=args.negative_prompt,
            use_auto_caption=args.use_auto_caption
        )
        
        # Process dataset
        success = augmenter.process_dataset(
            input_dir=args.input_dir,
            output_dir=output_dir,
            strength=args.strength,
            suffix=args.suffix,
            num_inference_steps=args.num_inference_steps
        )
        
        # Cleanup
        augmenter.cleanup()
        
        if success:
            print(f"\n🎉 Dataset augmentation completed successfully!")
            print(f"Augmented data saved to: {output_dir}")
            return 0
        else:
            print(f"\n❌ Dataset augmentation failed!")
            return 1
            
    except Exception as e:
        print(f"Error during augmentation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
