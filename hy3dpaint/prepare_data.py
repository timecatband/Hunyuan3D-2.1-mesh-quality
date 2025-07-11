#!/usr/bin/env python3
"""
Data preparation script for fine-tuning Hunyuan3D-Paint model on albedo map generation.

This script takes a textured mesh and generates training data in the format expected by
the HunyuanPaintPipeline for fine-tuning albedo map production.

Usage:
    python prepare_data.py --input_mesh path/to/mesh.obj --output_dir path/to/output --num_views 6
"""

import os
import sys
import json
import argparse
import numpy as np
# Fix numpy infinity issue
np.infty = np.inf
import torch
import trimesh
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import cv2
import math
import random

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import pyrender: {e}")
    print("Pyrender is required for textured rendering. Install with: pip install pyrender")
    PYRENDER_AVAILABLE = False

try:
    from DifferentiableRenderer.MeshRender import MeshRender
    from DifferentiableRenderer.mesh_utils import load_mesh
    from utils.uvwrap_utils import mesh_uv_wrap
except ImportError as e:
    print(f"Warning: Could not import MeshRender or mesh_utils: {e}")
    print("Make sure the DifferentiableRenderer is properly compiled and available")


class AlbedoDataPreparer:
    """
    Prepares training data for albedo map fine-tuning from a textured mesh.
    
    This class renders multiple views of a textured mesh and extracts the necessary
    conditioning information (normal maps, position maps) while preserving the
    ground truth albedo information for training.
    """
    
    def __init__(self, 
                 render_resolution: int = 512,
                 texture_resolution: int = 1024,
                 device: str = "cuda"):
        """
        Initialize the data preparer.
        
        Args:
            render_resolution: Resolution for rendered view images
            texture_resolution: Resolution for UV texture maps
            device: Device to use for rendering (cuda/cpu)
        """
        self.render_resolution = render_resolution
        self.texture_resolution = texture_resolution
        self.device = device
        self.mesh_renderer = None  # For normal/position rendering
        self.pyrender_scene = None  # For albedo/textured rendering
        self.pyrender_renderer = None
        self.mesh = None
        
    def create_look_at_pose(self, eye, target=None, up=None):
        """
        Creates a camera pose matrix (4x4) to look at a target.
        """
        if target is None:
            target = [0, 0, 0]
        if up is None:
            up = [0, 1, 0]
            
        eye = np.array(eye, dtype=np.float64)
        target = np.array(target, dtype=np.float64)
        up = np.array(up, dtype=np.float64)

        fwd = eye - target
        fwd /= np.linalg.norm(fwd)

        right = np.cross(up, fwd)
        right /= np.linalg.norm(right)

        new_up = np.cross(fwd, right)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = new_up
        pose[:3, 2] = fwd
        pose[:3, 3] = eye
        
        return pose
        
    def load_mesh(self, mesh_path: str) -> bool:
        """
        Load a textured mesh for processing.
        
        Args:
            mesh_path: Path to the mesh file (.obj, .ply, etc.)
            
        Returns:
            True if mesh loaded successfully, False otherwise
        """
        try:
            print(f"Loading mesh from: {mesh_path}")
            
            # Load mesh using trimesh
            self.mesh = trimesh.load(mesh_path)
            print("Loaded mesh with trimesh")
            
            if hasattr(self.mesh, 'visual') and hasattr(self.mesh.visual, 'material'):
                # Extract texture if available
                if hasattr(self.mesh.visual.material, 'image'):
                    print("Found texture in mesh")
                else:
                    print("Warning: No texture found in mesh")
            
            # Initialize MeshRender for normal/position rendering
            # CRITICAL: Use EXACT same settings as inference pipeline (textureGenPipeline.py)
            self.mesh_renderer = MeshRender(
                default_resolution=2048,        # Match inference: 1024 * 2
                texture_size=4096,              # Match inference: 1024 * 4  
                bake_mode="back_sample",        # Match inference
                raster_mode="cr",               # Match inference
            )
            
            # Load mesh into MeshRenderer for normal/position maps
            if isinstance(self.mesh, trimesh.Scene):
                mesh_for_renderer = self.mesh.dump(concatenate=True)
            else:
                mesh_for_renderer = self.mesh
            
            self.mesh_renderer.load_mesh(mesh=mesh_for_renderer)
            
            # Setup pyrender scene for albedo/textured rendering
            self._setup_pyrender_scene()
                
            print(f"Mesh loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading mesh: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_pyrender_scene(self):
        """Setup pyrender scene for textured rendering."""
        if not PYRENDER_AVAILABLE:
            print("Warning: Pyrender not available. Textured rendering will use fallback method.")
            return
        mesh = self.mesh.dump(concatenate=True) if isinstance(self.mesh, trimesh.Scene) else self.mesh
        
        pyrender_vertices = mesh.vertices.copy()
        max_bb = pyrender_vertices.max(axis=0)
        min_bb = pyrender_vertices.min(axis=0)
        center  = (max_bb + min_bb) / 2.0                             # ❶ bbox centre
        radius  = np.linalg.norm(pyrender_vertices - center, axis=1).max()
        scale   = (1.15 / (radius * 2.0)) if radius > 0 else 1.0      # ❷ identical formula

        # build transform:  translate to origin using the *same* centre, then scale
        transform = (
            trimesh.transformations.scale_matrix(scale) @
            trimesh.transformations.translation_matrix(-center)
        )

        mesh.apply_transform(transform)


        # Apply centering and scaling to the original mesh object
        # Note: The center for the transform should be from the original vertices
        #original_center = mesh.vertices.mean(axis=0) if not isinstance(mesh, trimesh.Scene) else mesh.bounds.mean(axis=0)
        #transform = trimesh.transformations.translation_matrix(-original_center)
        #transform = np.dot(trimesh.transformations.scale_matrix(scale), transform)
        #mesh.apply_transform(transform)

        try:
            # Center geometry using the entire scene's bounds
            if isinstance(mesh, trimesh.Scene):
                # The transformation is already applied to the scene's geometries
                scene_for_pyrender = mesh
            else:
                # The transformation is already applied to the mesh
                # Create a scene from single mesh
                scene_for_pyrender = trimesh.Scene([mesh])
            
            # Clean problematic textures (following render_glb.py pattern)
            geoms = scene_for_pyrender.geometry.values() if isinstance(scene_for_pyrender.geometry, dict) else [scene_for_pyrender.geometry]
            for geom in geoms:
                if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                    mat = geom.visual.material
                    if hasattr(mat, 'image') and isinstance(mat.image, np.ndarray):
                        mat.image = None
                    for attr in ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'occlusionTexture', 'emissiveTexture']:
                        if hasattr(mat, attr) and isinstance(getattr(mat, attr), np.ndarray):
                            setattr(mat, attr, None)
            
            # Create pyrender scene preserving valid textures

            self.pyrender_scene = pyrender.Scene.from_trimesh_scene(scene_for_pyrender, bg_color=[1.0, 1.0, 1.0, 0.0])
            
            # Add camera - Use Orthographic to match MeshRender
            ortho_scale = 1.2  # Match MeshRender's default ortho_scale
            camera = pyrender.OrthographicCamera(xmag=ortho_scale / 2.0, ymag=ortho_scale / 2.0, znear=0.1, zfar=100.0)
            self.camera_node = self.pyrender_scene.add(camera, pose=np.eye(4))
            
            # Add directional light
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            self.light_node = self.pyrender_scene.add(light, pose=np.eye(4))
            
            # Add ambient light to prevent completely black areas
            ambient_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
            ambient_pose = np.array([
                [0, 0, -1, 0],
                [0, 1, 0, 0], 
                [1, 0, 0, 0],
                [0, 0, 0, 1]
            ])
            self.ambient_node = self.pyrender_scene.add(ambient_light, pose=ambient_pose)
            
            # Initialize offscreen renderer
            self.pyrender_renderer = pyrender.OffscreenRenderer(
                viewport_width=self.render_resolution, 
                viewport_height=self.render_resolution
            )
            
            # Compute camera distance
            bbox_radius = scene_for_pyrender.bounding_sphere.primitive.radius
            fov = np.pi / 3.0
            self.base_camera_distance = 1.5
            
            print("Pyrender scene setup complete")
            
        except Exception as e:
            print(f"Error setting up pyrender scene: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_camera_poses(self, 
                            num_views: int = 6,
                            elevation_range: Tuple[float, float] = (-30, 30),
                            distance_range: Tuple[float, float] = (1.5, 2.5)) -> List[Dict]:
        """
        Generate camera poses for multi-view rendering, following MeshRender conventions.
        
        Args:
            num_views: Number of views to generate
            elevation_range: Range of elevation angles in degrees
            distance_range: Range of camera distances
            
        Returns:
            List of camera pose dictionaries
        """
        poses = []
        
        # Use the same default view configuration as Hunyuan3DPaintConfig
        base_azims = [0, 90, 180, 270, 0, 180]
        base_elevs = [0, 0, 0, 0, 90, -90]
        
        for i in range(num_views):
            if i < len(base_azims):
                # Use predefined views for first 6
                azimuth = base_azims[i]
                elevation = base_elevs[i]
            else:
                # Distribute additional azimuth angles evenly around the object
                azimuth = (360.0 / (num_views - 6)) * (i - 6)
                elevation = random.uniform(elevation_range[0], elevation_range[1])
            
            # Fixed distance for consistent results
            distance = 1.5
            
            pose = {
                'azimuth': azimuth,
                'elevation': elevation, 
                'distance': distance,
                'center': [0.0, 0.0, 0.0]  # Focus on origin
            }
            poses.append(pose)
            
        return poses
    
    def render_view(self, 
                   pose: Dict,
                   render_albedo: bool = True,
                   render_normal: bool = True,
                   render_position: bool = True) -> Dict[str, np.ndarray]:
        """
        Render a single view with different map types.
        
        Args:
            pose: Camera pose dictionary
            render_albedo: Whether to render albedo map
            render_normal: Whether to render normal map  
            render_position: Whether to render position map
            
        Returns:
            Dictionary containing rendered maps
        """
        if self.mesh_renderer is None or self.pyrender_scene is None:
            raise RuntimeError("No mesh loaded. Call load_mesh() first.")
        
        rendered_data = {}
        
        # Render RGB/Albedo view using pyrender
        if render_albedo:
            if self.pyrender_scene is not None and PYRENDER_AVAILABLE:
                # Setup camera pose for pyrender
                height_offset = 0.5 * (self.base_camera_distance * 0.2) * np.sin(np.radians(pose['azimuth'] * 2))
                eye = [
                    self.base_camera_distance * np.sin(np.radians(pose['azimuth'])), 
                    height_offset + self.base_camera_distance * np.sin(np.radians(pose['elevation'])), 
                    self.base_camera_distance * np.cos(np.radians(pose['azimuth']))
                ]
                camera_pose = self.create_look_at_pose(eye=eye, target=[0, 0, 0])
                self.pyrender_scene.set_pose(self.camera_node, pose=camera_pose)
                
                # Render albedo (no lighting effects)
                self.light_node.light.intensity = 0.0
                self.ambient_node.light.intensity = 0.0
                albedo_flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT
                albedo_color, _ = self.pyrender_renderer.render(self.pyrender_scene, flags=albedo_flags)
                
                # Restore lighting for potential other renders
                self.light_node.light.intensity = 3.0
                self.ambient_node.light.intensity = 1.5
                
                # Convert to [0, 1] range including alpha channel for transparency
                rgba_image = albedo_color / 255.0
                rendered_data['albedo'] = rgba_image
            else:
                # Fallback: Create a simple white albedo map
                print("Warning: Using fallback albedo rendering (white texture)")
                rgb_image = np.ones((self.render_resolution, self.render_resolution, 3), dtype=np.float32)
                rendered_data['albedo'] = rgb_image
        
        # CRITICAL FIX: Use EXACT same coordinate system as inference pipeline
        # The inference pipeline uses these camera poses exactly as defined in textureGenPipeline.py
        # and processes them through ViewProcessor.render_normal_multiview() -> MeshRender.render_normal()
        
        # Render normal map using MeshRender - MATCH INFERENCE SETTINGS EXACTLY
        if render_normal:
            # CRITICAL: Use the EXACT SAME parameters as inference pipeline (pipeline_utils.py)
            # The inference calls: self.render.render_normal(elev, azim, use_abs_coor=True, return_type="pl")
            normal_image = self.mesh_renderer.render_normal(
                elev=pose['elevation'],     # Use elevation AS-IS (get_mv_matrix will apply -elev internally)
                azim=pose['azimuth'],       # Use azimuth AS-IS (get_mv_matrix will apply +90 internally)
                camera_distance=pose.get('distance', 1.5),  # Use same distance as inference
                use_abs_coor=True,          # CRITICAL: Must match inference setting
                normalize_rgb=True,         # CRITICAL: Must match inference setting  
                return_type="np"            # Convert PIL to numpy for consistency
            )
            rendered_data['normal'] = normal_image
        
        # Render position map using MeshRender - MATCH INFERENCE SETTINGS EXACTLY  
        if render_position:
            # CRITICAL: Use the EXACT SAME parameters as inference pipeline (pipeline_utils.py)
            # The inference calls: self.render.render_position(elev, azim, return_type="pl")
            position_image = self.mesh_renderer.render_position(
                elev=pose['elevation'],     # Use elevation AS-IS (get_mv_matrix will apply -elev internally)
                azim=pose['azimuth'],       # Use azimuth AS-IS (get_mv_matrix will apply +90 internally) 
                camera_distance=pose.get('distance', 1.5),  # Use same distance as inference
                return_type="np"            # Convert PIL to numpy for consistency
            )
            rendered_data['position'] = position_image
        
        return rendered_data
    
    def generate_lighting_variations(self, 
                                   base_rendered_data: Dict[str, np.ndarray],
                                   pose: Dict) -> Dict[str, np.ndarray]:
        """
        Generate different lighting condition variations for conditioning images.
        Uses random camera angles and lighting angles with distance jitter.
        
        Args:
            base_rendered_data: Base rendered data dictionary (not used for camera pose)
            pose: Original camera pose dictionary (used only for fallback)
            
        Returns:
            Dictionary with different lighting variations from random viewpoints
        """
        # Generate lighting variations
        lighting_data = {}
        
        if self.pyrender_scene is None:
            # Fallback to simple variations if pyrender is not available
            # Use only RGB channels for lighting variations
            albedo_data = base_rendered_data['albedo'][..., :3]
            
            # Ambient lighting (darker)
            ambient_lit = albedo_data * 0.7
            lighting_data['light_AL'] = np.clip(ambient_lit, 0, 1)
            
            # Point lighting (brighter, more contrast)
            point_lit = np.power(albedo_data, 0.8)
            lighting_data['light_PL'] = np.clip(point_lit, 0, 1)
            
            # Environment map lighting (base image)
            lighting_data['light_ENVMAP'] = albedo_data
            
            return lighting_data
        
        self.pyrender_scene.ambient_light = [0.4, 0.4, 0.4, 1.0]  # Default ambient light
        
        # Helper function to create random camera pose with distance jitter
        def create_random_camera_pose():
            # Random azimuth and elevation for camera
            camera_azimuth = random.uniform(0, 360)
            camera_elevation = random.uniform(-30, 45)  # Reasonable elevation range
            
            # Add distance jitter
            distance_jitter = random.uniform(0.8, 1.4)  # 20% jitter up/down
            camera_distance = self.base_camera_distance * distance_jitter
            
            # Convert to cartesian coordinates
            height_offset = 0.5 * (camera_distance * 0.2) * np.sin(np.radians(camera_azimuth * 2))
            camera_pos = np.array([
                camera_distance * np.sin(np.radians(camera_azimuth)),
                height_offset + camera_distance * np.sin(np.radians(camera_elevation)),
                camera_distance * np.cos(np.radians(camera_azimuth))
            ])
            
            return self.create_look_at_pose(eye=camera_pos, target=[0, 0, 0])
        
        # Helper function to create random light pose
        def create_random_light_pose(base_distance_multiplier=1.0):
            # Random azimuth and elevation for lighting
            light_azimuth = random.uniform(0, 360)
            light_elevation = random.uniform(-45, 60)  # Avoid extreme low angles
            
            # Add small jitter to distance
            distance_jitter = random.uniform(0.8, 1.2) * base_distance_multiplier
            light_distance = self.base_camera_distance * distance_jitter
            
            # Convert to cartesian coordinates
            light_pos = np.array([
                light_distance * np.cos(np.radians(light_elevation)) * np.sin(np.radians(light_azimuth)),
                light_distance * np.sin(np.radians(light_elevation)),
                light_distance * np.cos(np.radians(light_elevation)) * np.cos(np.radians(light_azimuth))
            ])
            
            # Create look-at pose for light (pointing toward origin)
            return self.create_look_at_pose(eye=light_pos, target=[0, 0, 0])
        
        lit_flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        
        # Ambient lighting (low intensity ambient only) - Random camera angle
        random_camera_pose = create_random_camera_pose()
        self.pyrender_scene.set_pose(self.camera_node, pose=random_camera_pose)
        random_ambient_pose = create_random_light_pose(0.5)  # Closer for ambient
        self.pyrender_scene.set_pose(self.light_node, pose=random_ambient_pose)
        self.light_node.light.intensity = 0.5  # Lower intensity for ambient
        self.ambient_node.light.intensity = 2.0
        ambient_color, _ = self.pyrender_renderer.render(self.pyrender_scene, flags=lit_flags)
        lighting_data['light_AL'] = ambient_color[:, :, :3] / 255.0
        
        # Point lighting (strong directional light from random angle) - Random camera angle
        random_camera_pose = create_random_camera_pose()
        self.pyrender_scene.set_pose(self.camera_node, pose=random_camera_pose)
        random_point_pose = create_random_light_pose(1.0)
        self.pyrender_scene.set_pose(self.light_node, pose=random_point_pose)
        self.light_node.light.intensity = 5.0
        self.ambient_node.light.intensity = 0.9
        point_color, _ = self.pyrender_renderer.render(self.pyrender_scene, flags=lit_flags)
        lighting_data['light_PL'] = point_color[:, :, :3] / 255.0
        
        # Environment map lighting (balanced lighting from random angle) - Random camera angle
        random_camera_pose = create_random_camera_pose()
        self.pyrender_scene.set_pose(self.camera_node, pose=random_camera_pose)
        random_env_pose = create_random_light_pose(1.2)
        self.pyrender_scene.set_pose(self.light_node, pose=random_env_pose)
        self.light_node.light.intensity = 3.0
        self.ambient_node.light.intensity = 1.5
        env_color, _ = self.pyrender_renderer.render(self.pyrender_scene, flags=lit_flags)
        lighting_data['light_ENVMAP'] = env_color[:, :, :3] / 255.0
        
        # Key lighting (strong light from random key angle) - Random camera angle
        random_camera_pose = create_random_camera_pose()
        self.pyrender_scene.set_pose(self.camera_node, pose=random_camera_pose)
        random_key_pose = create_random_light_pose(0.8)  # Closer for more dramatic effect
        self.pyrender_scene.set_pose(self.light_node, pose=random_key_pose)
        self.light_node.light.intensity = 4.0
        self.ambient_node.light.intensity = 1.0
        key_color, _ = self.pyrender_renderer.render(self.pyrender_scene, flags=lit_flags)
        lighting_data['light_KEY'] = key_color[:, :, :3] / 255.0
        
        # Restore default lighting setup
        self.pyrender_scene.set_pose(self.light_node, pose=np.eye(4))
        self.light_node.light.intensity = 3.0
        self.ambient_node.light.intensity = 1.5
        
        return lighting_data
    
    def save_rendered_data(self, 
                          rendered_data: Dict[str, np.ndarray],
                          lighting_data: Dict[str, np.ndarray],
                          view_idx: int,
                          output_dir: Path) -> None:
        """
        Save rendered data to files in the expected format.
        
        Args:
            rendered_data: Main rendered data (albedo, normal, position)
            lighting_data: Lighting variation data
            view_idx: View index for naming
            output_dir: Output directory path
        """
        # Create subdirectories
        render_tex_dir = output_dir / "render_tex"
        render_cond_dir = output_dir / "render_cond"
        render_tex_dir.mkdir(exist_ok=True)
        render_cond_dir.mkdir(exist_ok=True)
        
        # Save main rendered data
        for map_type, data in rendered_data.items():
            if map_type == 'albedo':
                # Save RGB image
                filename = f"{view_idx:03d}.png"
                self._save_image(data, render_tex_dir / filename)
                
                # Save albedo map
                filename = f"{view_idx:03d}_albedo.png"
                self._save_image(data, render_tex_dir / filename)
                
            elif map_type == 'normal':
                filename = f"{view_idx:03d}_normal.png"
                self._save_image(data, render_tex_dir / filename)
                
            elif map_type == 'position':
                filename = f"{view_idx:03d}_pos.png"
                self._save_image(data, render_tex_dir / filename)
        
        # Save lighting condition variations
        for light_type, data in lighting_data.items():
            filename = f"{view_idx:03d}_{light_type}.png"
            self._save_image(data, render_cond_dir / filename)
        
        # Create placeholder metallic-roughness map (since we're only doing albedo)
        # In a real implementation, you would extract this from PBR materials
        albedo_rgb = rendered_data['albedo'][..., :3] if rendered_data['albedo'].shape[-1] == 4 else rendered_data['albedo']
        mr_data = np.ones_like(albedo_rgb)
        mr_data[:, :, 0] = 0.1  # Low metallic
        mr_data[:, :, 1] = 0.8  # Medium roughness
        mr_data[:, :, 2] = 0.0  # Unused channel
        
        mr_filename = f"{view_idx:03d}_mr.png"
        self._save_image(mr_data, render_tex_dir / mr_filename)
    
    def _save_image(self, image_data: np.ndarray, filepath: Path) -> None:
        """
        Save image data to file.
        
        Args:
            image_data: Image data as numpy array [H, W, C] in range [0, 1]
            filepath: Path to save the image
        """
        # Convert to 8-bit
        image_8bit = (np.clip(image_data, 0, 1) * 255).astype(np.uint8)
        
        # Convert to PIL and save
        if image_8bit.shape[-1] == 1:
            # Grayscale
            image_pil = Image.fromarray(image_8bit.squeeze(), mode='L')
        elif image_8bit.shape[-1] == 4:
            # RGBA with transparency
            image_pil = Image.fromarray(image_8bit, mode='RGBA')
        else:
            # RGB
            image_pil = Image.fromarray(image_8bit, mode='RGB')
        
        image_pil.save(filepath)
        print(f"Saved: {filepath}")
    
    def generate_camera_transforms(self, poses: List[Dict], output_dir: Path) -> None:
        """
        Generate camera transforms JSON file matching the expected format.
        
        The key insight is that MeshRender uses a specific coordinate system transformation:
        - elev = -elev (negates elevation)  
        - azim += 90 (adds 90° to azimuth)
        
        And the transforms.json expects camera-to-world matrices, not world-to-camera.
        
        Args:
            poses: List of camera poses
            output_dir: Output directory
        """
        transforms = {
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "scale": 0.5,
            "offset": [0.0, 0.0, 0.0],
            "frames": []
        }
        
        for i, pose in enumerate(poses):
            # Use MeshRender's get_mv_matrix function to get the world-to-camera matrix
            # then invert it to get the camera-to-world matrix that transforms.json expects
            from DifferentiableRenderer.camera_utils import get_mv_matrix
            
            elev = pose['elevation'] 
            azim = pose['azimuth']
            distance = pose['distance']
            
            # Get world-to-camera matrix using MeshRender's coordinate system
            w2c = get_mv_matrix(elev, azim, distance, center=[0, 0, 0])
            
            # Convert to camera-to-world matrix by inverting
            c2w = np.linalg.inv(w2c)
            
            # The JSON format expects azimuth/elevation in the ORIGINAL coordinate system
            # but adjusted by MeshRender's transformations for consistency
            # Based on the example data: azim=0° becomes azim=-90° in JSON
            azim_json = math.radians(azim - 90)  # Subtract 90° to match expected system
            elev_json = math.radians(elev)       # Elevation stays the same in JSON
            
            frame = {
                "file_path": f"{i:03d}.png",
                "camera_angle_x": math.radians(53.13),  # Standard FOV
                "proj_type": 1,
                "azimuth": azim_json,
                "elevation": elev_json,
                "cam_dis": distance,
                "transform_matrix": c2w.tolist()
            }
            transforms["frames"].append(frame)
            
            # Debug output for first frame to verify coordinate system
            if i == 0:
                print(f"Debug: First frame coordinate system check:")
                print(f"  Original pose: azim={azim}°, elev={elev}°")
                print(f"  JSON azimuth: {math.degrees(azim_json):.1f}°")
                print(f"  Transform matrix (c2w):")
                for row_idx in range(3):
                    row = c2w[row_idx]
                    print(f"    [{row_idx}]: [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}]")
                
                # Verify against expected pattern for front view
                if abs(azim) < 1e-6 and abs(elev) < 1e-6:  # This should be front view  
                    print(f"  Expected for front view (original azim=0°, JSON azim=-90°):")
                    print(f"    [0]: [1.0, 0.0, ~0, ~0]")
                    print(f"    [1]: [~0, ~0, -1.0, -1.5]") 
                    print(f"    [2]: [0.0, 1.0, ~0, 0.0]")
        
        # Save transforms
        transforms_path = output_dir / "render_tex" / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        print(f"Saved camera transforms: {transforms_path}")
    
    def create_dataset_json(self, output_dir: Path, sample_name: str) -> None:
        """
        Create the dataset JSON file for training.
        
        Args:
            output_dir: Output directory containing the sample
            sample_name: Name of the sample
        """
        dataset_entry = str(output_dir.absolute())
        
        json_data = [dataset_entry]
        
        json_path = output_dir.parent / f"{sample_name}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Created dataset JSON: {json_path}")
    
    def process_mesh(self,
                    input_mesh_path: str,
                    output_dir: str,
                    num_views: int = 6,
                    sample_name: Optional[str] = None) -> bool:
        """
        Process a single mesh and generate training data.
        
        Args:
            input_mesh_path: Path to input mesh
            output_dir: Directory to save output data
            num_views: Number of views to render
            sample_name: Name for the sample (auto-generated if None)
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            if sample_name is None:
                sample_name = Path(input_mesh_path).stem
            
            output_path = Path(output_dir) / sample_name
            output_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing mesh: {input_mesh_path}")
            print(f"Output directory: {output_path}")
            
            # Load mesh
            if not self.load_mesh(input_mesh_path):
                return False
            
            # Generate camera poses
            poses = self.generate_camera_poses(num_views)
            print(f"Generated {len(poses)} camera poses")
            
            # Render views
            for i, pose in enumerate(poses):
                print(f"Rendering view {i+1}/{len(poses)}")
                
                # Render main data
                rendered_data = self.render_view(pose)
                
                # Generate lighting variations
                lighting_data = self.generate_lighting_variations(rendered_data, pose)
                
                # Save data
                self.save_rendered_data(rendered_data, lighting_data, i, output_path)
            
            # Generate camera transforms
            self.generate_camera_transforms(poses, output_path)
            
            # Create dataset JSON
            self.create_dataset_json(output_path, sample_name)
            
            # Validate consistency with inference pipeline
            self.validate_inference_consistency(output_path)
            
            print(f"Successfully processed mesh. Training data saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error processing mesh: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Ensure cleanup happens
            self.cleanup()

    def cleanup(self):
        """Clean up rendering resources."""
        if self.pyrender_renderer is not None:
            self.pyrender_renderer.delete()
            self.pyrender_renderer = None
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
    
    def validate_inference_consistency(self, output_dir: Path) -> bool:
        """
        Validate that the prepared data is consistent with inference pipeline expectations.
        
        Args:
            output_dir: Directory containing prepared data
            
        Returns:
            True if validation passes, False if issues detected
        """
        print("🔍 Validating data consistency with inference pipeline...")
        
        render_tex_dir = output_dir / "render_tex"
        if not render_tex_dir.exists():
            print("❌ render_tex directory not found")
            return False
            
        # Check for required files
        normal_files = list(render_tex_dir.glob("*_normal.png"))
        position_files = list(render_tex_dir.glob("*_pos.png"))
        
        if len(normal_files) == 0:
            print("❌ No normal map files found")
            return False
            
        if len(position_files) == 0:
            print("❌ No position map files found")
            return False
            
        # Sample check: load a normal map and verify coordinate system
        try:
            from PIL import Image
            import numpy as np
            
            sample_normal = Image.open(normal_files[0])
            normal_array = np.array(sample_normal) / 255.0
            
            # Check if normal map uses absolute coordinates (as expected by inference)
            # Absolute coordinate normals should have more variation and different distribution
            normal_mean = np.mean(normal_array)
            normal_std = np.std(normal_array)
            
            print(f"✅ Normal map statistics - Mean: {normal_mean:.3f}, Std: {normal_std:.3f}")
            
            # Additional checks could be added here
            if normal_std < 0.1:
                print("⚠️  Warning: Normal map appears to have low variation - check use_abs_coor setting")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not validate normal map: {e}")
            
        print("✅ Data validation completed")
        return True

    # ...existing code...
def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Prepare training data for Hunyuan3D-Paint albedo fine-tuning")
    
    # Create mutually exclusive group for input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_mesh", type=str,
                           help="Path to input textured mesh (.obj, .ply, etc.)")
    input_group.add_argument("--input_mesh_list", type=str,
                           help="Path to text file containing list of mesh paths (one per line)")
    
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save training data")
    parser.add_argument("--num_views", type=int, default=6,
                       help="Number of views to render (default: 6)")
    parser.add_argument("--render_resolution", type=int, default=512,
                       help="Resolution for rendered images (default: 512)")
    parser.add_argument("--texture_resolution", type=int, default=1024,
                       help="Resolution for UV textures (default: 1024)")
    parser.add_argument("--sample_name", type=str, default=None,
                       help="Name for the sample (default: mesh filename). Only used with --input_mesh")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for rendering (default: cuda)")
    
    args = parser.parse_args()
    
    # Get list of mesh files to process
    mesh_files = []
    if args.input_mesh:
        # Single mesh file
        if not os.path.exists(args.input_mesh):
            print(f"Error: Input mesh file not found: {args.input_mesh}")
            return 1
        mesh_files = [args.input_mesh]
    else:
        # Multiple mesh files from list
        if not os.path.exists(args.input_mesh_list):
            print(f"Error: Input mesh list file not found: {args.input_mesh_list}")
            return 1
        
        try:
            with open(args.input_mesh_list, 'r') as f:
                mesh_files = [line.strip() for line in f if line.strip()]
            
            # Validate that all mesh files exist
            missing_files = [f for f in mesh_files if not os.path.exists(f)]
            if missing_files:
                print(f"Error: The following mesh files were not found:")
                for f in missing_files:
                    print(f"  - {f}")
                return 1
                
            print(f"Found {len(mesh_files)} mesh files to process")
        except Exception as e:
            print(f"Error reading mesh list file: {e}")
            return 1
    
    # Create data preparer
    preparer = AlbedoDataPreparer(
        render_resolution=args.render_resolution,
        texture_resolution=args.texture_resolution,
        device=args.device
    )
    
    # Process each mesh file
    successful_count = 0
    failed_count = 0
    
    for i, mesh_file in enumerate(mesh_files):
        print(f"\n{'='*60}")
        print(f"Processing mesh {i+1}/{len(mesh_files)}: {mesh_file}")
        print(f"{'='*60}")

        if i > 500:
            print("Skipping further processing after 500 meshes")
            sys.exit(0)
        
        # Determine sample name
        if args.input_mesh and args.sample_name:
            # Single mesh with custom name
            sample_name = args.sample_name
        else:
            # Use mesh filename as sample name
            sample_name = Path(mesh_file).stem
        
        # Process mesh
        success = preparer.process_mesh(
            input_mesh_path=mesh_file,
            output_dir=args.output_dir,
            num_views=args.num_views,
            sample_name=sample_name
        )
        
        if success:
            successful_count += 1
            print(f"✓ Successfully processed: {mesh_file}")
        else:
            failed_count += 1
            print(f"✗ Failed to process: {mesh_file}")
    
    # Cleanup
    preparer.cleanup()
    
    # Final summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total meshes: {len(mesh_files)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    
    if successful_count > 0:
        print(f"\nTraining data saved to: {args.output_dir}")
        print("You can now use this data for fine-tuning the albedo generation model.")
    
    if failed_count == 0:
        print("\n🎉 All meshes processed successfully!")
        return 0
    elif successful_count > 0:
        print(f"\n⚠️  {successful_count} meshes processed successfully, {failed_count} failed.")
        return 0
    else:
        print("\n❌ All meshes failed to process!")
        return 1


if __name__ == "__main__":
    sys.exit(main())