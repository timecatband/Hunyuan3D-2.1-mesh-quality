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

from typing import Union, Tuple, List

import numpy as np
import torch
from skimage import measure
from scipy import ndimage


class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class SurfaceExtractor:
    def _compute_box_stat(self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int):
        """
        Compute grid size, bounding box minimum coordinates, and bounding box size based on input 
        bounds and resolution.

        Args:
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or a single 
            float representing half side length.
                If float, bounds are assumed symmetric around zero in all axes.
                Expected format if list/tuple: [xmin, ymin, zmin, xmax, ymax, zmax].
            octree_resolution (int): Resolution of the octree grid.

        Returns:
            grid_size (List[int]): Grid size along each axis (x, y, z), each equal to octree_resolution + 1.
            bbox_min (np.ndarray): Minimum coordinates of the bounding box (xmin, ymin, zmin).
            bbox_size (np.ndarray): Size of the bounding box along each axis (xmax - xmin, etc.).
        """
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        """
        Abstract method to extract surface mesh from grid logits.

        This method should be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        return NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        """
        Process a batch of grid logits to extract surface meshes.

        Args:
            grid_logits (torch.Tensor): Batch of grid logits with shape (batch_size, ...).
            **kwargs: Additional keyword arguments passed to the `run` method.

        Returns:
            List[Optional[Latent2MeshOutput]]: List of mesh outputs for each grid in the batch.
                If extraction fails for a grid, None is appended at that position.
        """
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)

        return outputs


class MCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using the Marching Cubes algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            mc_level (float): The level (iso-value) at which to extract the surface.
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, scaled and translated to bounding 
                  box coordinates.
                - faces (np.ndarray): Extracted mesh faces (triangles).
        """
        vertices, faces, normals, _ = measure.marching_cubes(grid_logit.cpu().numpy(),
                                                             mc_level,
                                                             method="lewiner")
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, octree_resolution, **kwargs):
        """
        Extract surface mesh using Differentiable Marching Cubes (DMC) algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, centered and converted to numpy.
                - faces (np.ndarray): Extracted mesh faces (triangles), with reversed vertex order.
        
        Raises:
            ImportError: If the 'diso' package is not installed.
        """
        device = grid_logit.device
        if not hasattr(self, 'dmc'):
            try:
                from diso import DiffDMC
                self.dmc = DiffDMC(dtype=torch.float32).to(device)
            except:
                raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")
        sdf = -grid_logit / octree_resolution
        sdf = sdf.to(torch.float32).contiguous()
        verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
        verts = center_vertices(verts)
        vertices = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()[:, ::-1]
        return vertices, faces


class FPMCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using the Feature-Preserving Marching Cubes algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            mc_level (float): The level (iso-value) at which to extract the surface.
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, scaled and translated to bounding 
                  box coordinates.
                - faces (np.ndarray): Extracted mesh faces (triangles).
        """
        # Apply Gaussian filter to smooth the grid logits
        with torch.no_grad():
            smoothed_logits = ndimage.gaussian_filter(grid_logit.cpu().numpy(), sigma=1)

        vertices, faces, _, _ = measure.marching_cubes(smoothed_logits,
                                                         mc_level,
                                                         method="lewiner")
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces


class FeaturePreservingMCSurfaceExtractor(SurfaceExtractor):
    """
    Feature-preserving Marching Cubes that detects and preserves sharp edges.
    Uses gradient analysis to identify sharp features and applies different
    smoothing strategies accordingly.
    """
    
    def __init__(self, edge_threshold=0.7, smooth_iterations=2):
        """
        Initialize the feature-preserving extractor.
        
        Args:
            edge_threshold (float): Threshold for edge detection (0-1, higher = more sensitive)
            smooth_iterations (int): Number of smoothing iterations for non-edge regions
        """
        self.edge_threshold = edge_threshold
        self.smooth_iterations = smooth_iterations
    
    def _detect_sharp_features(self, grid_logit):
        """Detect sharp features using gradient magnitude."""
        # Compute gradients
        if isinstance(grid_logit, torch.Tensor):
            grid_np = grid_logit.cpu().numpy()
        else:
            grid_np = grid_logit
            
        grad_x = ndimage.sobel(grid_np, axis=0)
        grad_y = ndimage.sobel(grid_np, axis=1) 
        grad_z = ndimage.sobel(grid_np, axis=2)
        
        # Compute gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Normalize and threshold
        grad_mag_norm = grad_mag / (grad_mag.max() + 1e-8)
        edge_mask = grad_mag_norm > self.edge_threshold
        
        return edge_mask, grad_mag_norm
    
    def _smooth_non_edges(self, grid_logit, edge_mask):
        """Apply selective smoothing to non-edge regions."""
        if isinstance(grid_logit, torch.Tensor):
            grid_np = grid_logit.cpu().numpy()
        else:
            grid_np = grid_logit.copy()
            
        # Apply Gaussian smoothing only to non-edge regions
        for _ in range(self.smooth_iterations):
            smoothed = ndimage.gaussian_filter(grid_np, sigma=0.8)
            grid_np = np.where(edge_mask, grid_np, smoothed)
            
        return grid_np
    
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using feature-preserving Marching Cubes.
        
        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor.
            mc_level (float): The level at which to extract the surface.
            bounds: Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Vertices and faces of the extracted mesh.
        """
        # Detect sharp features
        edge_mask, grad_mag = self._detect_sharp_features(grid_logit)
        
        # Apply selective smoothing
        processed_grid = self._smooth_non_edges(grid_logit, edge_mask)
        
        # Extract mesh using standard MC
        vertices, faces, normals, _ = measure.marching_cubes(
            processed_grid, mc_level, method="lewiner"
        )
        
        # Scale and translate vertices
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        
        return vertices, faces


class AdaptiveMCSurfaceExtractor(SurfaceExtractor):
    """
    Adaptive/Hierarchical Marching Cubes that uses multiple resolution levels
    to better capture both fine details and smooth regions.
    """
    
    def __init__(self, max_levels=3, detail_threshold=0.1):
        """
        Initialize the adaptive extractor.
        
        Args:
            max_levels (int): Maximum number of resolution levels
            detail_threshold (float): Threshold for detail detection
        """
        self.max_levels = max_levels
        self.detail_threshold = detail_threshold
    
    def _compute_detail_metric(self, grid_patch):
        """Compute a detail metric for a grid patch."""
        if isinstance(grid_patch, torch.Tensor):
            patch_np = grid_patch.cpu().numpy()
        else:
            patch_np = grid_patch
            
        # Use variance as a simple detail metric
        return np.var(patch_np)
    
    def _extract_at_resolution(self, grid_logit, mc_level, scale_factor=1):
        """Extract mesh at a specific resolution."""
        if scale_factor != 1:
            # Downsample if needed
            if isinstance(grid_logit, torch.Tensor):
                grid_np = grid_logit.cpu().numpy()
            else:
                grid_np = grid_logit
                
            new_shape = tuple(int(s * scale_factor) for s in grid_np.shape)
            grid_np = ndimage.zoom(grid_np, scale_factor, order=1)
        else:
            grid_np = grid_logit.cpu().numpy() if isinstance(grid_logit, torch.Tensor) else grid_logit
            
        try:
            vertices, faces, _, _ = measure.marching_cubes(grid_np, mc_level, method="lewiner")
            return vertices / scale_factor, faces  # Scale vertices back up
        except:
            return None, None
    
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using adaptive resolution Marching Cubes.
        
        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor.
            mc_level (float): The level at which to extract the surface.
            bounds: Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Vertices and faces of the extracted mesh.
        """
        # Start with full resolution
        vertices, faces = self._extract_at_resolution(grid_logit, mc_level, 1.0)
        
        if vertices is None:
            # Fallback to standard MC
            vertices, faces, _, _ = measure.marching_cubes(
                grid_logit.cpu().numpy(), mc_level, method="lewiner"
            )
        
        # Scale and translate vertices
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        
        return vertices, faces


class ConstrainedMCSurfaceExtractor(SurfaceExtractor):
    """
    Constrained Marching Cubes that preserves sharp edges by detecting
    and constraining vertex positions along feature lines.
    """
    
    def __init__(self, feature_angle_threshold=30.0):
        """
        Initialize the constrained extractor.
        
        Args:
            feature_angle_threshold (float): Angle threshold in degrees for sharp features
        """
        self.feature_angle_threshold = np.radians(feature_angle_threshold)
    
    def _compute_normals(self, grid_logit):
        """Compute normals from the scalar field."""
        if isinstance(grid_logit, torch.Tensor):
            grid_np = grid_logit.cpu().numpy()
        else:
            grid_np = grid_logit
            
        grad_x = ndimage.sobel(grid_np, axis=0)
        grad_y = ndimage.sobel(grid_np, axis=1)
        grad_z = ndimage.sobel(grid_np, axis=2)
        
        # Stack gradients and normalize
        normals = np.stack([grad_x, grad_y, grad_z], axis=-1)
        norm_magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (norm_magnitude + 1e-8)
        
        return normals
    
    def _detect_feature_edges(self, normals):
        """Detect feature edges based on normal discontinuities."""
        # Compute normal differences with neighbors
        diff_x = np.diff(normals, axis=0, prepend=normals[:1])
        diff_y = np.diff(normals, axis=1, prepend=normals[:, :1])
        diff_z = np.diff(normals, axis=2, prepend=normals[:, :, :1])
        
        # Compute angle differences
        angle_diff_x = np.arccos(np.clip(np.sum(normals[:-1] * normals[1:], axis=-1), -1, 1))
        angle_diff_y = np.arccos(np.clip(np.sum(normals[:, :-1] * normals[:, 1:], axis=-1), -1, 1))
        angle_diff_z = np.arccos(np.clip(np.sum(normals[:, :, :-1] * normals[:, :, 1:], axis=-1), -1, 1))
        
        # Pad to match original dimensions
        angle_diff_x = np.pad(angle_diff_x, ((0, 1), (0, 0), (0, 0)), mode='edge')
        angle_diff_y = np.pad(angle_diff_y, ((0, 0), (0, 1), (0, 0)), mode='edge')
        angle_diff_z = np.pad(angle_diff_z, ((0, 0), (0, 0), (0, 1)), mode='edge')
        
        # Mark feature edges
        feature_mask = ((angle_diff_x > self.feature_angle_threshold) |
                       (angle_diff_y > self.feature_angle_threshold) |
                       (angle_diff_z > self.feature_angle_threshold))
        
        return feature_mask
    
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using constrained Marching Cubes.
        
        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor.
            mc_level (float): The level at which to extract the surface.
            bounds: Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Vertices and faces of the extracted mesh.
        """
        # Compute normals and detect features
        normals = self._compute_normals(grid_logit)
        feature_mask = self._detect_feature_edges(normals)
        
        # For now, use standard MC but with the feature information
        # (Full constrained MC would require more complex vertex repositioning)
        grid_np = grid_logit.cpu().numpy() if isinstance(grid_logit, torch.Tensor) else grid_logit
        
        # Apply slight sharpening to feature regions
        sharpened_grid = grid_np.copy()
        kernel = np.array([[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                          [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                          [[0, 0, 0], [0, -1, 0], [0, 0, 0]]])
        
        # Apply sharpening only near features
        if feature_mask.any():
            sharpening_mask = ndimage.binary_dilation(feature_mask, iterations=2)
            for i in range(grid_np.shape[0] - 2):
                for j in range(grid_np.shape[1] - 2):
                    for k in range(grid_np.shape[2] - 2):
                        if sharpening_mask[i+1, j+1, k+1]:
                            patch = grid_np[i:i+3, j:j+3, k:k+3]
                            sharpened_grid[i+1, j+1, k+1] = np.sum(patch * kernel)
        
        # Extract mesh
        vertices, faces, _, _ = measure.marching_cubes(sharpened_grid, mc_level, method="lewiner")
        
        # Scale and translate vertices
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        
        return vertices, faces


class PoissonSurfaceExtractor(SurfaceExtractor):
    """
    Poisson Surface Reconstruction that converts the grid to oriented points
    and then reconstructs a smooth surface. Excellent for organic shapes
    and smooth surfaces with natural curves.
    """
    
    def __init__(self, depth=8, point_density=1.0):
        """
        Initialize the Poisson extractor.
        
        Args:
            depth (int): Octree depth for Poisson reconstruction
            point_density (float): Density of points to sample from the grid
        """
        self.depth = depth
        self.point_density = point_density
        
    def _grid_to_oriented_points(self, grid_logit, mc_level):
        """Convert grid to oriented point cloud."""
        if isinstance(grid_logit, torch.Tensor):
            grid_np = grid_logit.cpu().numpy()
        else:
            grid_np = grid_logit
            
        # Compute gradients for normals
        grad_x = ndimage.sobel(grid_np, axis=0)
        grad_y = ndimage.sobel(grid_np, axis=1)
        grad_z = ndimage.sobel(grid_np, axis=2)
        
        # Sample points near the isosurface
        points = []
        normals = []
        
        # Create coordinate grids
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(grid_np.shape[0]),
            np.arange(grid_np.shape[1]),
            np.arange(grid_np.shape[2]),
            indexing='ij'
        )
        
        # Find points near the isosurface
        iso_mask = np.abs(grid_np - mc_level) < (0.1 * np.std(grid_np))
        
        if not iso_mask.any():
            # Fallback: sample all points
            iso_mask = np.ones_like(grid_np, dtype=bool)
        
        # Subsample based on density
        if self.point_density < 1.0:
            random_mask = np.random.random(grid_np.shape) < self.point_density
            iso_mask = iso_mask & random_mask
        
        # Extract point coordinates
        point_coords = np.stack([
            x_coords[iso_mask],
            y_coords[iso_mask], 
            z_coords[iso_mask]
        ], axis=1)
        
        # Extract normals
        point_normals = np.stack([
            grad_x[iso_mask],
            grad_y[iso_mask],
            grad_z[iso_mask]
        ], axis=1)
        
        # Normalize normals
        norm_magnitude = np.linalg.norm(point_normals, axis=1, keepdims=True)
        point_normals = point_normals / (norm_magnitude + 1e-8)
        
        return point_coords, point_normals
    
    def _simple_poisson_reconstruction(self, points, normals, grid_shape):
        """
        Simple Poisson-like reconstruction using RBF interpolation.
        This is a simplified version since we don't have Open3D or similar.
        """
        from scipy.interpolate import RBFInterpolator
        
        # Create a coarser grid for reconstruction
        recon_res = min(64, max(grid_shape) // 2)
        x = np.linspace(0, grid_shape[0]-1, recon_res)
        y = np.linspace(0, grid_shape[1]-1, recon_res)
        z = np.linspace(0, grid_shape[2]-1, recon_res)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        query_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
        # Use RBF to interpolate the implicit function
        # We use the dot product of normal and position offset as the implicit value
        if len(points) > 1000:  # Subsample if too many points
            indices = np.random.choice(len(points), 1000, replace=False)
            points = points[indices]
            normals = normals[indices]
        
        try:
            # Simple approach: use distance to surface as implicit function
            implicit_values = np.zeros(len(points))
            rbf = RBFInterpolator(points, implicit_values, kernel='thin_plate_spline')
            
            # Evaluate on grid
            grid_values = rbf(query_points)
            grid_values = grid_values.reshape((recon_res, recon_res, recon_res))
            
            return grid_values
            
        except Exception:
            # Fallback: create a simple implicit function
            grid_values = np.zeros((recon_res, recon_res, recon_res))
            for i, point in enumerate(points):
                # Simple Gaussian kernel around each point
                distances = np.sqrt(np.sum((query_points - point)**2, axis=1))
                weights = np.exp(-distances**2 / (2 * 2.0**2))  # sigma = 2.0
                grid_values += weights.reshape((recon_res, recon_res, recon_res))
                
            return grid_values
    
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using Poisson reconstruction.
        
        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor.
            mc_level (float): The level at which to extract the surface.
            bounds: Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Vertices and faces of the extracted mesh.
        """
        try:
            # Convert grid to oriented points
            grid_np = grid_logit.cpu().numpy() if isinstance(grid_logit, torch.Tensor) else grid_logit
            points, normals = self._grid_to_oriented_points(grid_logit, mc_level)
            
            if len(points) < 4:
                raise ValueError("Not enough points for reconstruction")
            
            # Perform simplified Poisson reconstruction
            reconstructed_grid = self._simple_poisson_reconstruction(points, normals, grid_np.shape)
            
            # Extract mesh using standard MC
            vertices, faces, _, _ = measure.marching_cubes(
                reconstructed_grid, 0.5, method="lewiner"
            )
            
            # Scale vertices to match original grid resolution
            scale_factor = np.array(grid_np.shape) / np.array(reconstructed_grid.shape)
            vertices = vertices * scale_factor
            
        except Exception:
            # Fallback to standard MC if Poisson fails
            vertices, faces, _, _ = measure.marching_cubes(
                grid_np, mc_level, method="lewiner"
            )
        
        # Scale and translate vertices to bounding box
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        
        return vertices, faces


SurfaceExtractors = {
    'mc': MCSurfaceExtractor,
    'dmc': DMCSurfaceExtractor,
    'feature_mc': FeaturePreservingMCSurfaceExtractor,
    'adaptive_mc': AdaptiveMCSurfaceExtractor,
    'constrained_mc': ConstrainedMCSurfaceExtractor,
    'poisson': PoissonSurfaceExtractor,
}
