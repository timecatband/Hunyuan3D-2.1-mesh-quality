from PIL import Image

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
import sys
from peft import LoraConfig, get_peft_model, PeftModel
import torch

model_path = 'tencent/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, octree_resolution=128)
pipeline_shapegen.model.eval()

image_path = sys.argv[1] if len(sys.argv) > 1 else 'demo_image.png'
image = Image.open(image_path).convert("RGBA")
#if image.mode == 'RGB':
rembg = BackgroundRemover()
image = rembg(image)


print("cats")
peft_model = PeftModel.from_pretrained(pipeline_shapegen.model, 'lora_adapters')
pipeline_shapegen.model = peft_model

# Load the extra conditional token
cond_tok = torch.load('lora_adapters/cond_token.pt')
print("Conditional token shape:", cond_tok.shape)
#cond_tok = None
#print("Not inserting extra token")

mesh = pipeline_shapegen(image=image, num_inference_steps=50, guidance_scale=15.0, extra_cond_token=cond_tok, classifier_scale=0.0, octree_resolution=378)[0]
mesh.export('demo_4.glb')


