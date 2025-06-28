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


