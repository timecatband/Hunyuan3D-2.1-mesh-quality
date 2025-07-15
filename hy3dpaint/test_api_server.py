#!/usr/bin/env python3
"""
Test script for the Hunyuan LORA API server
"""

import requests
import base64
import json
import time
import sys
from pathlib import Path
import uuid

def test_api_server(host="localhost", port=8083):
    """Test the API server with a simple request"""
    
    base_url = f"http://{host}:{port}"
    
    # Test health check
    print("Testing health check...")
    try:
        response = requests.get(f"{base_url}/healthcheck")
        if response.status_code == 200:
            print("✓ Health check passed")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Test worker status
    print("Testing worker status...")
    try:
        response = requests.get(f"{base_url}/worker_status")
        if response.status_code == 200:
            status = response.json()
            print(f"✓ Worker status: {status}")
        else:
            print(f"✗ Worker status failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Worker status failed: {e}")
    
    print("API server tests completed")
    return True

def create_test_obj():
    """Create a simple test OBJ file"""
    obj_content = """# Simple cube
v -1.0 -1.0  1.0
v  1.0 -1.0  1.0
v  1.0  1.0  1.0
v -1.0  1.0  1.0
v -1.0 -1.0 -1.0
v  1.0 -1.0 -1.0
v  1.0  1.0 -1.0
v -1.0  1.0 -1.0

vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0

f 1/1 2/2 3/3 4/4
f 2/1 6/2 7/3 3/4
f 6/1 5/2 8/3 7/4
f 5/1 1/2 4/3 8/4
f 4/1 3/2 7/3 8/4
f 1/1 5/2 6/3 2/4
"""
    return obj_content.encode('utf-8')

def create_test_image():
    """Create a simple test image"""
    from PIL import Image
    import io
    
    # Create a simple colored square
    img = Image.new('RGB', (256, 256), color='red')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_generation(host="localhost", port=8083, mesh_path=None, image_path=None):
    """Test the generation endpoint (blocking)"""
    
    base_url = f"http://{host}:{port}"
    
    # Prepare test data (use provided files or defaults)
    if mesh_path:
        obj_data = Path(mesh_path).read_bytes()
    else:
        obj_data = create_test_obj()
    if image_path:
        img_data = Path(image_path).read_bytes()
    else:
        img_data = create_test_image()
    
    # Encode as base64
    obj_b64 = base64.b64encode(obj_data).decode('utf-8')
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    
    # Prepare request
    payload = {
        "obj_model": obj_b64,
        "image": img_b64,
        "geo_rotation": 0,
        "diffusion_steps": 10,  # Use fewer steps for testing
        "no_rembg": True,
        "use_pbr": False
    }
    
    print("Testing generation endpoint...")
    try:
        # Use blocking /generate endpoint
        response = requests.post(f"{base_url}/generate", json=payload)
        if response.status_code == 200:
            uid_hex = uuid.uuid4().hex
            out_path = Path(f"{uid_hex}.glb")
            out_path.write_bytes(response.content)
            print(f"✓ Generation completed, saved to {out_path.resolve()}")
            return True
        else:
            print(f"✗ Generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Hunyuan LORA API server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8083, help="Server port")
    parser.add_argument("--test-generation", action="store_true", 
                       help="Test generation endpoint (requires working model)")
    parser.add_argument("--mesh", help="Path to mesh OBJ file")
    parser.add_argument("--image", help="Path to image file")
    
    args = parser.parse_args()
    
    # Basic API tests
    if not test_api_server(args.host, args.port):
        sys.exit(1)
    
    # Generation test (optional)
    if args.test_generation:
        if not test_generation(args.host, args.port, mesh_path=args.mesh, image_path=args.image):
            print("Generation test failed, but basic API is working")
        else:
            print("All tests passed!")
    else:
        print("Basic API tests passed! Use --test-generation to test the full pipeline")
    