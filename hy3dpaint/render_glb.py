import os
import glob
import numpy as np
np.infty = np.inf
import sys
import trimesh
import pyrender
from PIL import Image

def create_look_at_pose(eye, target=None, up=None):
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

def render_glb_from_angles(glb_path, num_angles=8, output_dir='renderings'):
    """
    Renders a GLB file from multiple angles, producing both lit and albedo renderings.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if output file already exists
    base_name = os.path.splitext(os.path.basename(glb_path))[0]
    grid_path = os.path.join(output_dir, f"{base_name}.png")
    if os.path.exists(grid_path):
        print(f"Output file {grid_path} already exists, skipping {glb_path}")
        return

    # Initialize lists to collect lit and albedo images for grid
    lit_images = []
    albedo_images = []

    # --- Load and prepare scene ---
    try:
        scene = trimesh.load(glb_path, force='scene')
    except Exception as e:
        print(f"Error loading GLB file: {e}")
        return
    # Center geometry using the entire scene's bounds
    center = scene.bounds.mean(axis=0)
    transform = trimesh.transformations.translation_matrix(-center)
    scene.apply_transform(transform)

    # Compute camera distance to fit object in viewport
    # Use bounding sphere radius for consistent framing
    bbox_radius = scene.bounding_sphere.primitive.radius
    
    fov = np.pi / 3.0
    # Distance to fit sphere with a small margin to prevent clipping
    # Account for aspect ratio (viewport is 640x480, so height is limiting factor)
    aspect_ratio = 640 / 480
    effective_fov = fov if aspect_ratio >= 1.0 else fov / aspect_ratio
    
    # Use tangent for more accurate distance calculation
    base_camera_distance = bbox_radius / np.tan(effective_fov / 2) * 1.8
    # Clean only numpy-array textures
    geoms = scene.geometry.values() if isinstance(scene.geometry, dict) else [scene.geometry]
    for geom in geoms:
        if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
            mat = geom.visual.material
            if hasattr(mat, 'image') and isinstance(mat.image, np.ndarray):
                mat.image = None
            for attr in ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'occlusionTexture', 'emissiveTexture']:
                if hasattr(mat, attr) and isinstance(getattr(mat, attr), np.ndarray):
                    setattr(mat, attr, None)
    # Create pyrender scene preserving valid textures
    pyrender_scene = pyrender.Scene.from_trimesh_scene(scene, bg_color=[0.8, 0.8, 0.8, 1.0])

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_node = pyrender_scene.add(camera, pose=np.eye(4))

    # Add directional light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_node = pyrender_scene.add(light, pose=np.eye(4))
    
    # Add ambient light to prevent completely black areas
    ambient_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    ambient_pose = np.array([
        [0, 0, -1, 0],
        [0, 1, 0, 0], 
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    ambient_node = pyrender_scene.add(ambient_light, pose=ambient_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

    # --- Rendering Loop ---
    for i in range(num_angles):
        angle = 2 * np.pi * i / num_angles
        # Position camera around object with more significant height variation
        height_offset = 0.5 * bbox_radius * np.sin(angle * 2)  # More height variation
        eye = [
            base_camera_distance * np.sin(angle), 
            height_offset, 
            base_camera_distance * np.cos(angle)
        ]
        camera_pose = create_look_at_pose(eye=eye, target=[0, 0, 0])
        pyrender_scene.set_pose(camera_node, pose=camera_pose)

        # --- Lit Rendering ---
        lit_flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        lit_color, _ = renderer.render(pyrender_scene, flags=lit_flags)

        # --- Albedo Rendering (raw textures, no lighting) ---
        # Temporarily disable lights
        light_node.light.intensity = 0.0
        ambient_node.light.intensity = 0.0
        albedo_flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT
        albedo_color, _ = renderer.render(pyrender_scene, flags=albedo_flags)
        # Restore light intensity
        light_node.light.intensity = 3.0
        ambient_node.light.intensity = 1.5

        # --- Save Renderings ---
        lit_image = Image.fromarray(lit_color)
        albedo_image = Image.fromarray(albedo_color)

        lit_image.save(os.path.join(output_dir, f"lit_render_{i}.png"))
        albedo_image.save(os.path.join(output_dir, f"albedo_render_{i}.png"))
        # Append to lists for combined grid
        lit_images.append(lit_image)
        albedo_images.append(albedo_image)

        print(f"Rendered angle {i+1}/{num_angles}")

    renderer.delete()

    # Combine all renderings into a single grid image (lit in first row, albedo in second)
    if lit_images:
        # Determine grid size with spacing
        w, h = lit_images[0].size
        cols = len(lit_images)
        rows = 2
        spacing = 10  # pixels between images
        grid_width = cols * w + (cols - 1) * spacing
        grid_height = rows * h + (rows - 1) * spacing
        grid_img = Image.new('RGB', (grid_width, grid_height), color=(200, 200, 200))
        
        # Paste lit images in first row
        for idx, img in enumerate(lit_images):
            x_pos = idx * (w + spacing)
            y_pos = 0
            grid_img.paste(img, (x_pos, y_pos))
        
        # Paste albedo images in second row
        for idx, img in enumerate(albedo_images):
            x_pos = idx * (w + spacing)
            y_pos = h + spacing
            grid_img.paste(img, (x_pos, y_pos))
        base_name = os.path.splitext(os.path.basename(glb_path))[0]
        grid_path = os.path.join(output_dir, f"{base_name}.png")
        grid_img.save(grid_path)
        print(f"Saved grid image at {grid_path}")

    print("\nDone!")

if __name__ == '__main__':
    # Process single file or directory of GLBs
    if len(sys.argv) < 2:
        print("Usage: python render_glb.py <input_path> [output_dir]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'renderings'
    number_of_renders = 3

    if not os.path.exists(input_path):
        print(f"Error: The path '{input_path}' was not found.")
    elif os.path.isdir(input_path):
        # Render all GLB files in directory
        glb_files = glob.glob(os.path.join(input_path, '*.glb'))
        if not glb_files:
            print(f"No GLB files found in directory '{input_path}'.")
        else:
            for glb in sorted(glb_files):
                try:
                    render_glb_from_angles(glb, num_angles=number_of_renders, output_dir=output_dir)
                except Exception as e:
                    print(f"Error rendering {glb}: {e}")
    else:
        # Single GLB file
        render_glb_from_angles(input_path, num_angles=number_of_renders, output_dir=output_dir)
    print("\nAll done!")
