#!/usr/bin/env python3
"""
Example script for running LORA training on Hunyuan3D-Paint.

This script demonstrates how to use the train_lora.py script with sample data.
"""

import os
import sys
import json
from pathlib import Path

# Example usage function
def create_example_dataset_json():
    """Create an example dataset JSON file."""
    # This should point to your prepared training data
    example_data = [
        "/path/to/your/training/data/sample1",
        "/path/to/your/training/data/sample2",
        # Add more paths as needed
    ]
    
    output_file = "example_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(example_data, f, indent=2)
    
    print(f"Created example dataset JSON: {output_file}")
    return output_file

def run_lora_training():
    """Run LORA training with example parameters."""
    
    # Create example dataset JSON
    dataset_json = create_example_dataset_json()
    
    # Training parameters
    cmd = [
        "python", "train_lora.py",
        "--dataset_json", dataset_json,
        "--output_dir", "lora_checkpoints",
        "--batch_size", "1",
        "--learning_rate", "1e-4", 
        "--num_epochs", "50",
        "--save_every", "10",
        "--lora_rank", "16",
        "--lora_alpha", "32",
        "--lora_dropout", "0.1",
        "--gradient_accumulation_steps", "4",
        "--max_grad_norm", "1.0",
        # "--use_wandb",  # Uncomment to use Weights & Biases logging
        # "--wandb_project", "my-hunyuan3d-lora",
    ]
    
    print("Running LORA training with command:")
    print(" ".join(cmd))
    print("\nTo run manually, use the command above or:")
    print(f"python train_lora.py --dataset_json {dataset_json} --output_dir lora_checkpoints")

def create_config_template():
    """Create a template configuration file."""
    config = {
        "multiview_pretrained_path": "tencent/Hunyuan3D-2.1",
        "custom_pipeline": "hunyuanpaintpbr",
        "training": {
            "batch_size": 1,
            "learning_rate": 1e-4,
            "num_epochs": 50,
            "save_every": 10,
            "gradient_accumulation_steps": 4,
            "mixed_precision": True,
            "max_grad_norm": 1.0
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_k", "to_q", "to_v", "to_out.0",
                "proj_in", "proj_out", 
                "ff.net.0.proj", "ff.net.2"
            ]
        }
    }
    
    with open("lora_config_template.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created LORA configuration template: lora_config_template.json")

if __name__ == "__main__":
    print("Hunyuan3D-Paint LORA Training Example")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "config":
        create_config_template()
    else:
        run_lora_training()
        
    print("\nNext steps:")
    print("1. Install LORA training dependencies:")
    print("   pip install -r requirements_lora.txt")
    print("\n2. Prepare your training data using prepare_data.py:")
    print("   python prepare_data.py --input_mesh_list mesh_list.txt --output_dir training_data")
    print("\n3. Update the dataset paths in example_dataset.json")
    print("\n4. Run the training:")
    print("   python train_lora.py --dataset_json example_dataset.json --output_dir lora_checkpoints")
    print("\n5. Use the trained LORA adapter for inference")
