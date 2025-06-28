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

from PIL import Image

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
import sys
import torch 
model_path = 'tencent/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, octree_resolution=128)

input_dir = sys.argv[1]
# List all images in the directory
images = []
import os
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGBA")
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)
        images.append(image)

i = 0
for image in images:
    print("Processing image...")
    # Generate mesh from image
    # TODO: Temporary unconditional
    latents = pipeline_shapegen(image=image, num_inference_steps=50, guidance_scale=0.0, classifier_scale = 0.0, output_type="latent")[0]
    torch.save(latents, 'output_latents' + str(i) + '.pt')
    i += 1


