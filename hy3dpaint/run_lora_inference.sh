#!/bin/bash

# --- Script to Run Python Inference on GLB/PNG Pairs ---
#
# This script accepts an input directory as an argument. It then iterates
# through each immediate subdirectory within that input directory.
#
# For each subdirectory, it expects to find exactly one .glb file and one .png file.
# It then constructs and executes a python command using those files as inputs.
#
# Usage:
# ./run_inference.sh /path/to/your/main_folder
#
# Example directory structure:
# /path/to/your/main_folder/
# ├── asset_one/
# │   ├── model.glb
# │   └── texture.png
# └── asset_two/
#     ├── another_model.glb
#     └── another_texture.png

# --- Argument Handling ---

# Check if an input directory was provided.
if [ -z "$1" ]; then
  echo "Error: No input directory specified."
  echo "Usage: $0 <input_directory>"
  exit 1
fi

INPUT_DIR="$1"

# Check if the provided input directory actually exists.
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Directory '$INPUT_DIR' not found."
  exit 1
fi

echo "Starting inference process for directories in '$INPUT_DIR'..."
echo "=========================================================="

# --- Main Loop ---

# Loop through each item in the input directory.
# The '*/' glob ensures we only process directories.
for dir in "$INPUT_DIR"/*/; do
    # Check if the found path is actually a directory to avoid errors.
    if [ ! -d "$dir" ]; then
        continue
    fi

    echo "Processing directory: $dir"

    # Find the .glb and .png files within the subdirectory.
    # We store them in arrays to easily check how many were found.
    # Using 'shopt -s nullglob' ensures that if no files match, the array is empty.
    shopt -s nullglob
    glb_files=("$dir"*.glb)
    png_files=("$dir"*.png)
    shopt -u nullglob # Turn nullglob off again to restore default behavior.

    # --- Validation ---
    # Check if we found exactly one of each file type.
    if [ ${#glb_files[@]} -ne 1 ]; then
        echo " -> Warning: Expected 1 GLB file, but found ${#glb_files[@]}. Skipping."
        echo "----------------------------------------------------------"
        continue
    fi

    if [ ${#png_files[@]} -ne 1 ]; then
        echo " -> Warning: Expected 1 PNG file, but found ${#png_files[@]}. Skipping."
        echo "----------------------------------------------------------"
        continue
    fi

    # --- Command Execution ---
    # Assign the full paths to variables for clarity.
    glb_path="${glb_files[0]}"
    png_path="${png_files[0]}"

    # Get the base name of the GLB file (without the .glb extension).
    base_name=$(basename "$glb_path" .glb)

    # Define the full path for the output file. It will be saved in the same subdirectory.
    output_path="${dir}${base_name}_textured.glb"

    echo " -> Found GLB: $glb_path"
    echo " -> Found PNG: $png_path"
    echo " -> Output will be: $output_path"
    echo " -> Executing Python script..."

    # Execute the command. Using quotes around variables ensures that
    # paths with spaces or special characters are handled correctly.
    python3 inference_lora.py \
        --lora_checkpoint lora_epoch_80 \
        --mesh_path "$glb_path" \
        --image_path "$png_path" \
        --output_mesh "$output_path"

    echo "----------------------------------------------------------"
done

echo "Inference process complete."
