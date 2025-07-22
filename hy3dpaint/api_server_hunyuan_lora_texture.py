#!/usr/bin/env python3
"""
Hunyuan3D-Paint LORA API Server
A FastAPI server that provides texture generation for 3D models using Hunyuan3D-Paint pipeline with LORA adapters.
API compatible with MVPainter but uses HunyuanPaint internally.
"""

import os
import sys
import json
import argparse
import logging
import logging.handlers
import uuid
import time
import threading
import asyncio
import uvicorn
import base64
import shutil
import subprocess
import traceback
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
import trimesh
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import gc

# Hunyuan3D-Paint imports
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

# LORA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT not available. Install with: pip install peft")
    PEFT_AVAILABLE = False

# Background removal support
try:
    import rembg
    REMBG_AVAILABLE = True
    
    # Try different possible locations for background removal utilities
    try:
        from src.utils.infer_util import remove_background, resize_foreground
    except ImportError:
        try:
            from utils.infer_util import remove_background, resize_foreground
        except ImportError:
            print("Warning: Background removal utilities not found, using fallbacks")
            
            def remove_background(image, session):
                """Fallback background removal using rembg directly"""
                if session is not None:
                    return rembg.remove(image, session=session)
                return image
            
            def resize_foreground(image, scale=0.85):
                """Fallback foreground resizing - simple center crop/pad"""
                return image
            
except ImportError:
    print("Warning: rembg not available")
    REMBG_AVAILABLE = False
    
    def remove_background(image, session):
        return image
    
    def resize_foreground(image, scale=0.85):
        return image

# Optional mesh reduction
MESH_REDUCTION_AVAILABLE = False
try:
    from scripts.remesh_reduce_blender_script import reduce_mesh
    MESH_REDUCTION_AVAILABLE = True
except ImportError:
    try:
        from remesh_reduce_blender_script import reduce_mesh
        MESH_REDUCTION_AVAILABLE = True
    except ImportError:
        print("Warning: Mesh reduction not available, using Blender decimation")
        MESH_REDUCTION_AVAILABLE = False

def reduce_mesh_blender(input_path, target_vertices=40000, output_path=None, blender_path="/usr/bin/blender"):
    """Reduce mesh using Blender decimation script"""
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_reduced.glb"
    
    script_path = os.path.join(os.path.dirname(__file__), "blender_single_decimation.py")
    
    cmd = [
        blender_path,
        "--background",
        "--python", script_path,
        "--",
        input_path,
        output_path,
        str(target_vertices)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info(f"Blender decimation successful: {output_path}")
            return output_path
        else:
            logger.error(f"Blender decimation failed: {result.stderr}")
            return input_path
    except subprocess.TimeoutExpired:
        logger.error("Blender decimation timed out")
        return input_path
    except Exception as e:
        logger.error(f"Blender decimation error: {e}")
        return input_path

# Apply torchvision fix if available
try:
    from utils.torchvision_fix import apply_fix
    apply_fix()
except Exception as e:
    print(f"Warning: Could not apply torchvision fix: {e}")

LOGDIR = '.'
SAVE_DIR = 'hunyuan_cache'
RENDER_TEMP_DIR = 'render_temp'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RENDER_TEMP_DIR, exist_ok=True)

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger("hunyuan_controller", f"{SAVE_DIR}/hunyuan_controller.log")

# Global variables
model_semaphore = None


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


class HunyuanLoraWorker:
    def __init__(self,
                 lora_checkpoint_path: Optional[str] = None,
                 blender_path: str = '/usr/bin/blender',
                 device: str = 'cuda',
                 seed: int = 12,
                 limit_model_concurrency: int = 5,
                 max_num_view: int = 6,
                 resolution: int = 512,
                 pbr_settings: List[str] = None):
        self.worker_id = worker_id
        self.device = device
        self.blender_path = blender_path
        self.seed = seed
        self.limit_model_concurrency = limit_model_concurrency
        self.lora_checkpoint_path = lora_checkpoint_path
        self.max_num_view = max_num_view
        self.resolution = resolution
        self.pbr_settings = pbr_settings or ["albedo"]  # Default to albedo-only
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        logger.info(f"Loading Hunyuan3D-Paint model on worker {worker_id} ...")
        
        # Initialize pipeline
        total_start_time = time.time()
        logger.info('Model loading started')
        
        self.setup_base_pipeline()
        self.load_lora_adapter()
        self.load_shading_token()
        
        # Initialize background remover if available
        if REMBG_AVAILABLE:
            self.rembg_session = rembg.new_session()
        else:
            self.rembg_session = None
        
        logger.info(f'Total model loading time: {time.time() - total_start_time:.2f} seconds')

    def setup_base_pipeline(self):
        """Setup the base Hunyuan3D-Paint pipeline."""
        logger.info("Setting up base pipeline...")
        
        # Create config
        conf = Hunyuan3DPaintConfig(self.max_num_view, self.resolution)
        
        # Create pipeline
        self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
        
        # Configure PBR settings
        self.paint_pipeline.set_active_pbr_settings(self.pbr_settings)
        
        logger.info("Base pipeline loaded successfully")
        
    def load_lora_adapter(self):
        """Load the trained LORA adapter if specified."""
        if self.lora_checkpoint_path is None:
            logger.info("No LORA checkpoint specified, using base model")
            return
            
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required for LORA inference. Install with: pip install peft")
            
        logger.info(f"Loading LORA adapter from: {self.lora_checkpoint_path}")
        
        # Get the inner UNet, which is the actual target for LoRA modifications
        inner_unet_main = self.paint_pipeline.models["multiview_model"].pipeline.unet
        
        # Load the LoRA weights into the inner UNet using PeftModel
        lora_unet_main = PeftModel.from_pretrained(inner_unet_main, self.lora_checkpoint_path, adapter_name="default")
        lora_unet_main.set_adapters("default", 1.0)
        
        # Merge the LoRA weights into the base model
        logger.info("Merging LoRA adapter into main generator UNet...")
        self.paint_pipeline.models["multiview_model"].pipeline.unet = lora_unet_main.merge_and_unload()
        
        logger.info("LORA adapter loaded successfully")
        del self.paint_pipeline.models["multiview_model"].pipeline.text_encoder
        
    def load_shading_token(self):
        """Load the learned albedo shading token if present."""
        if self.lora_checkpoint_path is None:
            self.learnable_shading_token = None
            return
            
        token_path = Path(self.lora_checkpoint_path) / "learnable_albedo_token.pt"
        if token_path.exists():
            self.learnable_shading_token = torch.load(token_path, map_location="cpu")
            logger.info(f"Loaded shading token from {token_path}")
        else:
            self.learnable_shading_token = None
            logger.info("No shading token found in checkpoint; proceeding without it")

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return self.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        """
        Main generation function that runs the Hunyuan3D-Paint pipeline
        Compatible with MVPainter API but uses HunyuanPaint internally
        """
        try:
            # Parse parameters (MVPainter compatible)
            if 'image' in params:
                # Base64 encoded reference image
                reference_image = load_image_from_base64(params['image'])
            else:
                raise ValueError("No reference image provided")

            # Extract mesh data
            if 'mesh' in params:
                model_data = base64.b64decode(params['mesh'])
                object_uid = str(uid)
                # Determine GLB magic ("glTF") or default to OBJ
                if model_data[:4] == b'glTF':
                    filename = f"{object_uid}.glb"
                else:
                    filename = f"{object_uid}.obj"
                obj_path = os.path.join(SAVE_DIR, filename)
                with open(obj_path, 'wb') as f:
                    f.write(model_data)
            else:
                raise ValueError("No mesh provided")

            # Parse optional parameters
            geo_rotation = params.get('geo_rotation', 0)
            diffusion_steps = params.get('diffusion_steps', 50)
            no_rembg = params.get('no_rembg', False)
            use_pbr = params.get('use_pbr', False)  # Not used in Hunyuan but kept for compatibility

            logger.info(f"Starting texture generation for {object_uid}")
            logger.info(f"Parameters: geo_rotation={geo_rotation}, steps={diffusion_steps}, no_rembg={no_rembg}")

            # Process reference image
            if not no_rembg and self.rembg_session is not None:
                reference_image = remove_background(reference_image, self.rembg_session)
                reference_image = resize_foreground(reference_image, 0.85)
            
            # Generate texture using Hunyuan3D-Paint pipeline
            result_path = self.run_hunyuan_pipeline(
                obj_path, reference_image, object_uid, geo_rotation
            )
            
            logger.info(f"Texture generation completed: {result_path}")
            return result_path, object_uid

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            raise

    def run_hunyuan_pipeline(self, obj_path, reference_image, object_uid, geo_rotation=0):
        """Run the Hunyuan3D-Paint pipeline to generate textured mesh"""
        
        # Create output directory
        output_dir = os.path.join(SAVE_DIR, 'results', object_uid)
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare output mesh path
        glb_output_dir = os.path.join(output_dir, 'glbs')
        os.makedirs(glb_output_dir, exist_ok=True)
        final_path = os.path.join(glb_output_dir, f'{object_uid}.glb')
        
        # Use Blender decimation to reduce mesh
        processed_obj_path = reduce_mesh_blender(
            obj_path, target_vertices=40000,
            output_path=os.path.join(output_dir, f'{object_uid}_reduced.glb'),
            blender_path=self.blender_path
        )

        # Save reference image temporarily
        ref_image_path = os.path.join(output_dir, 'reference.png')
        reference_image.save(ref_image_path)
        
        logger.info("Running Hunyuan3D-Paint texture generation...")
        
        # Display memory estimate
        estimated_vram = self.paint_pipeline.get_memory_usage_estimate()
        logger.info(f"Estimated VRAM usage: {estimated_vram:.1f} GB")
        self.paint_pipeline.set_active_pbr_settings(self.pbr_settings)
        # Generate texture using the Hunyuan3D-Paint pipeline
        result_path = self.paint_pipeline(
            mesh_path=processed_obj_path,
            image_path=ref_image_path,
            output_mesh_path=final_path,
            use_remesh=True,
            save_glb=True,
            learnable_shading_token=self.learnable_shading_token,
            pbr_settings=self.pbr_settings
        )
        
        # Clean up temporary files
        try:
            os.remove(ref_image_path)
            if processed_obj_path != obj_path:
                os.remove(processed_obj_path)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

        gc.collect()
        torch.cuda.empty_cache()
        
        return result_path


# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate_texture")
async def generate(request: Request):
    """Generate textured GLB from input mesh and reference image"""
    logger.info("Worker generating...")
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        logger.error("Caught ValueError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=400)
    except Exception as e:
        logger.error("Caught Unknown Error", e)
        traceback.print_exc()
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=500)


@app.post("/send")
async def send_generate(request: Request):
    """Start generation in background and return job ID"""
    logger.info("Worker send...")
    params = await request.json()
    uid = uuid.uuid4()
    threading.Thread(target=worker.generate, args=(uid, params,)).start()
    ret = {"uid": str(uid)}
    return JSONResponse(ret, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    """Check generation status and retrieve result"""
    # Check for GLB file in multiple possible locations
    possible_paths = [
        os.path.join(SAVE_DIR, 'results', uid, 'glbs', f'{uid}.glb'),
        os.path.join(SAVE_DIR, f'{uid}.glb'),
    ]
    
    save_file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            save_file_path = path
            break
    
    if save_file_path is None:
        response = {'status': 'processing'}
        return JSONResponse(response, status_code=200)
    else:
        with open(save_file_path, 'rb') as f:
            base64_str = base64.b64encode(f.read()).decode('utf-8')
        response = {'status': 'completed', 'model_base64': base64_str}
        return JSONResponse(response, status_code=200)


@app.get("/healthcheck")
async def healthcheck():
    """Health check endpoint"""
    return JSONResponse({"status": "ok"}, status_code=200)


@app.get("/worker_status")
async def worker_status():
    """Get worker status including queue length"""
    status = worker.get_status()
    return JSONResponse(status, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                       help="Path to LORA checkpoint directory")
    parser.add_argument("--blender_path", type=str, default="/usr/bin/blender")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--max_num_view", type=int, default=6,
                       help="Maximum number of views")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Output resolution")
    parser.add_argument("--pbr_settings", nargs="+", default=["albedo"],
                       choices=["albedo", "mr"],
                       help="PBR materials to generate (default: albedo only)")
    args = parser.parse_args()
    
    logger.info(f"args: {args}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = HunyuanLoraWorker(
        lora_checkpoint_path=args.lora_checkpoint,
        blender_path=args.blender_path,
        device=args.device,
        seed=args.seed,
        limit_model_concurrency=args.limit_model_concurrency,
        max_num_view=args.max_num_view,
        resolution=args.resolution,
        pbr_settings=args.pbr_settings
    )
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    logger.info(f"args: {args}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = HunyuanLoraWorker(
        lora_checkpoint_path=args.lora_checkpoint,
        blender_path=args.blender_path,
        device=args.device,
        seed=args.seed,
        limit_model_concurrency=args.limit_model_concurrency,
        max_num_view=args.max_num_view,
        resolution=args.resolution,
        pbr_settings=args.pbr_settings
    )
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
