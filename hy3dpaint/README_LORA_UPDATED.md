# Updated LORA Training for Hunyuan3D-Paint

This updated version of `train_lora.py` now properly loads the pretrained weights the same way `demo.py` does, using the Huggingface model repository and the custom pipeline.

## Key Changes

### 1. Proper Pretrained Weight Loading
- Now uses `huggingface_hub.snapshot_download()` to get the same pretrained weights as demo.py
- Loads from `"tencent/Hunyuan3D-2.1"` with pattern `"hunyuan3d-paintpbr-v2-1/*"`
- Uses the correct custom pipeline `"hunyuanpaintpbr"`
- Maintains the same model architecture and weights as the inference pipeline

### 2. Simplified Training Wrapper
- Created `TrainingWrapper` class that wraps the loaded pipeline
- Provides basic training methods compatible with LORA fine-tuning
- Focuses on albedo-only training for simplicity and efficiency
- Maintains compatibility with the dataset and PEFT integration

### 3. Streamlined Approach
- Removes complex model reconstruction and focuses on using the actual pretrained pipeline
- Simplified training step that focuses on core albedo texture generation
- Maintains proper noise scheduling and latent encoding from the original model

## Usage

### Basic Albedo Training (Recommended)
```bash
python train_lora.py \
    --dataset_json /path/to/dataset.json \
    --output_dir ./lora_outputs \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --lora_rank 16 \
    --lora_alpha 32
```

### With Weights & Biases Logging
```bash
python train_lora.py \
    --dataset_json /path/to/dataset.json \
    --use_wandb \
    --wandb_project my-hunyuan3d-lora \
    --batch_size 1 \
    --num_epochs 100
```

## Configuration Options

### Model Loading
- Automatically downloads and loads the same pretrained weights as demo.py
- Uses the official `tencent/Hunyuan3D-2.1` model from Huggingface
- Loads with the same custom pipeline and configuration

### LORA Configuration
- `--lora_rank`: LORA rank parameter (default: 16)
- `--lora_alpha`: LORA alpha parameter (default: 32)
- `--lora_dropout`: LORA dropout rate (default: 0.1)

### Training Configuration
- `--batch_size`: Training batch size (default: 1)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs`: Number of training epochs (default: 100)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)

## Technical Details

### Model Architecture
- Uses the exact same UNet2p5DConditionModel as the original pipeline
- Preserves all learned text clip tokens and specialized attention mechanisms
- Maintains compatibility with multiview processing and PBR material generation

### Training Approach
- Focuses on albedo texture generation (most important for visual quality)
- Uses proper noise scheduling from the pretrained DDPMScheduler
- Applies LORA adapters to all linear layers in the UNet
- Freezes VAE, text encoder, and other non-trainable components

### Memory Efficiency
- LORA adapters provide parameter-efficient fine-tuning
- Mixed precision training support for reduced memory usage
- Gradient accumulation for effective larger batch sizes

## Benefits

1. **Authentic Weights**: Uses the exact same pretrained weights as the original model
2. **Consistency**: Maintains full compatibility with the inference pipeline
3. **Efficiency**: LORA fine-tuning with minimal memory overhead
4. **Simplicity**: Streamlined training focused on core texture generation
5. **Reliability**: Based on the proven model loading approach from demo.py

## Notes

- The training wrapper currently focuses on albedo generation for simplicity
- MR (metallic-roughness) loss can be enabled but adds complexity
- The approach prioritizes compatibility and reliability over advanced training features
- Perfect for fine-tuning on custom texture datasets while preserving model capabilities
