"""
SDF utilities for physics-aware correction.

Provides efficient SDF computation and querying via Open3D's
RaycastingScene, plus helpers for mesh pre-processing (watertight
check, convex decomposition, etc.).
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple
import warnings


class MeshSDF:
    """
    Pre-built SDF for a mesh, cached for repeated queries during optimisation.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self._mesh = o3d.geometry.TriangleMesh(mesh)
        self._mesh.compute_vertex_normals()
        self._scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self._mesh)
        self._scene.add_triangles(mesh_t)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Query SDF at world-space points. Returns (N,) float32 array."""
        pts = o3d.core.Tensor(points.astype(np.float32))
        return self._scene.compute_signed_distance(pts).numpy()

    def closest_point(self, points: np.ndarray) -> np.ndarray:
        """Find closest surface point for each query."""
        pts = o3d.core.Tensor(points.astype(np.float32))
        res = self._scene.compute_closest_points(pts)
        return res['points'].numpy()

    @property
    def mesh(self):
        return self._mesh


def check_watertight(mesh: o3d.geometry.TriangleMesh) -> bool:
    """Heuristic watertight check."""
    edges = mesh.get_non_manifold_edges()
    return len(edges) == 0


def compute_convex_hull(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Compute convex hull for collision proxy."""
    pcd = mesh.sample_points_uniformly(number_of_points=4096)
    hull, _ = pcd.compute_convex_hull()
    return hull


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh,
                        num_points: int = 2048) -> np.ndarray:
    """Uniformly sample a mesh surface → (N, 3) numpy array."""
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)


# Alias for compatibility
compute_sdf_for_mesh_cached = MeshSDF
