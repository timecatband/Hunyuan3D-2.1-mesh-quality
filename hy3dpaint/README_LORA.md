# Hunyuan3D-Paint LORA Training

This directory contains tools for fine-tuning the Hunyuan3D-Paint model using LORA (Low-Rank Adaptation) for efficient parameter training.

## Overview

The LORA training system consists of several components:

1. **Data Preparation** (`prepare_data.py`) - Converts textured meshes into training data
2. **LORA Training** (`train_lora.py`) - Trains LORA adapters on the prepared data  
3. **Inference** (`inference_lora.py`) - Uses trained LORA adapters for texture generation
4. **Examples** (`example_lora_training.py`) - Example scripts and configurations

## Quick Start

### 1. Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install LORA training requirements
pip install -r requirements_lora.txt
```

### 2. Prepare Training Data

Create training data from textured meshes:

```bash
# Single mesh
python prepare_data.py --input_mesh path/to/mesh.obj --output_dir training_data --num_views 6

# Multiple meshes from a list
echo "path/to/mesh1.obj" > mesh_list.txt
echo "path/to/mesh2.obj" >> mesh_list.txt
python prepare_data.py --input_mesh_list mesh_list.txt --output_dir training_data --num_views 6
```

This creates a dataset structure:
```
training_data/
├── sample1/
│   ├── render_tex/
│   │   ├── 000.png, 000_albedo.png, 000_normal.png, 000_pos.png, 000_mr.png
│   │   ├── 001.png, 001_albedo.png, 001_normal.png, 001_pos.png, 001_mr.png
│   │   └── ...
│   ├── render_cond/
│   │   ├── 000_light_AL.png, 000_light_PL.png, 000_light_ENVMAP.png
│   │   └── ...
│   └── transforms.json
└── sample2/
    └── ...
```

### 3. Create Dataset JSON

Create a JSON file listing all training samples:

```json
[
  "/path/to/training_data/sample1",
  "/path/to/training_data/sample2"
]
```

### 4. Train LORA Adapter

```bash
python train_lora.py \
  --dataset_json dataset.json \
  --output_dir lora_checkpoints \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --num_epochs 50 \
  --save_every 10 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --gradient_accumulation_steps 4
```

### 5. Use Trained Model for Inference

```bash
python inference_lora.py \
  --lora_checkpoint lora_checkpoints/checkpoint-epoch-50 \
  --mesh_path path/to/test_mesh.obj \
  --image_path path/to/reference_image.jpg \
  --output_mesh textured_output.obj
```

## Detailed Usage

### Data Preparation Options

The `prepare_data.py` script supports various options:

```bash
python prepare_data.py \
  --input_mesh_list mesh_list.txt \
  --output_dir training_data \
  --num_views 6 \
  --render_resolution 512 \
  --texture_resolution 1024 \
  --device cuda
```

**Key Features:**
- Renders albedo, normal, and position maps
- Generates lighting condition variations (AL, PL, ENVMAP, KEY)
- Creates proper camera transform matrices
- Supports transparent backgrounds for albedo images
- Compatible with the training data loader format

### Training Configuration

Key training parameters:

- `--lora_rank`: LORA rank (lower = fewer parameters, higher = more capacity)
- `--lora_alpha`: LORA scaling parameter (typically 2x rank)
- `--lora_dropout`: Dropout rate for LORA layers
- `--gradient_accumulation_steps`: Accumulate gradients over multiple batches
- `--mixed_precision`: Use automatic mixed precision for efficiency
- `--use_wandb`: Enable Weights & Biases logging

### LORA Configuration

The LORA adapter targets these UNet modules:
- Attention layers: `to_k`, `to_q`, `to_v`, `to_out.0`
- Projection layers: `proj_in`, `proj_out`
- Feed-forward layers: `ff.net.0.proj`, `ff.net.2`

### Monitoring Training

With Weights & Biases enabled:

```bash
python train_lora.py \
  --dataset_json dataset.json \
  --output_dir lora_checkpoints \
  --use_wandb \
  --wandb_project my-hunyuan3d-project
```

This logs:
- Training loss (total, albedo, metallic-roughness)
- Learning rate schedule
- Training progress and metrics

### Memory Optimization

For training with limited GPU memory:

```bash
python train_lora.py \
  --dataset_json dataset.json \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lora_rank 8 \
  --mixed_precision
```

### Resuming Training

To resume from a checkpoint:

```bash
python train_lora.py \
  --dataset_json dataset.json \
  --resume_from lora_checkpoints/checkpoint-epoch-20 \
  --num_epochs 50
```

## File Structure

```
hy3dpaint/
├── prepare_data.py              # Data preparation script
├── train_lora.py               # LORA training script
├── inference_lora.py           # LORA inference script
├── example_lora_training.py    # Example usage
├── requirements_lora.txt       # Additional dependencies
├── textureGenPipeline.py      # Base pipeline
└── src/
    └── data/
        └── dataloader/
            └── objaverse_loader_forTexturePBR.py  # Data loader
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size to 1
   - Increase gradient accumulation steps
   - Use lower LORA rank
   - Enable mixed precision

2. **PEFT Import Errors**
   ```bash
   pip install peft>=0.7.0
   ```

3. **Data Format Issues**
   - Ensure `prepare_data.py` was used to create training data
   - Check that JSON file contains absolute paths
   - Verify all required image files exist

4. **Convergence Issues**
   - Try different learning rates (1e-5 to 1e-3)
   - Adjust LORA rank and alpha
   - Increase training epochs
   - Check data quality and diversity

### Performance Tips

1. **Training Speed**
   - Use mixed precision (`--mixed_precision`)
   - Optimize batch size for your GPU
   - Use gradient accumulation instead of large batches

2. **Quality**
   - Use diverse training data
   - Higher LORA rank for better capacity
   - Longer training with lower learning rate
   - Monitor validation metrics

3. **Memory Usage**
   - Lower batch size
   - Gradient checkpointing (built into model)
   - Lower resolution for training

## Example Workflows

### Basic Workflow

```bash
# 1. Prepare data
python prepare_data.py --input_mesh mesh.obj --output_dir data

# 2. Create dataset JSON
echo '["'$(pwd)'/data/mesh"]' > dataset.json

# 3. Train LORA
python train_lora.py --dataset_json dataset.json --num_epochs 30

# 4. Run inference
python inference_lora.py \
  --lora_checkpoint lora_checkpoints/checkpoint-epoch-30 \
  --mesh_path test_mesh.obj \
  --image_path ref_image.jpg
```

### Batch Processing Workflow

```bash
# 1. Prepare multiple meshes
find /path/to/meshes -name "*.obj" > mesh_list.txt
python prepare_data.py --input_mesh_list mesh_list.txt --output_dir batch_data

# 2. Create dataset JSON from all samples
find batch_data -type d -name "*" | grep -v "^batch_data$" | \
  python -c "import sys, json; print(json.dumps([line.strip() for line in sys.stdin]))" > batch_dataset.json

# 3. Train with wandb logging
python train_lora.py \
  --dataset_json batch_dataset.json \
  --use_wandb \
  --wandb_project hunyuan3d-batch \
  --num_epochs 100
```

This comprehensive setup provides efficient LORA fine-tuning for the Hunyuan3D-Paint model with proper data preparation, training, and inference capabilities.
