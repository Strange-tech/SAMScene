"""
Point cloud utilities: sampling, normalisation, registration helpers.
"""

import numpy as np
from typing import Optional, Tuple


def fps(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS).

    Args:
        points: (N, D) input point set
        num_samples: number of points to sample

    Returns:
        (num_samples, D) sampled points (indices in order)
    """
    n = len(points)
    if n <= num_samples:
        return points.copy()

    indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(n, np.inf, dtype=np.float32)

    # Start with a random point
    indices[0] = np.random.randint(n)
    farthest = points[indices[0]]

    for i in range(1, num_samples):
        dist = np.linalg.norm(points - farthest, axis=1)
        distances = np.minimum(distances, dist)
        indices[i] = np.argmax(distances)
        farthest = points[indices[i]]

    return points[indices]


def normalize_to_canonical(pc: np.ndarray,
                           target_range: Tuple[float, float] = (-1.0, 1.0)) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize point cloud to canonical [-1, 1]^3 space (isotropic).

    Returns:
        normalized_pc, center, scale
    """
    if len(pc) == 0:
        return pc, np.zeros(3), 1.0

    center = (pc.min(axis=0) + pc.max(axis=0)) / 2.0
    centered = pc - center
    max_dist = np.linalg.norm(centered, axis=1).max()
    if max_dist < 1e-6:
        max_dist = 1.0
    scale = max_dist
    lo, hi = target_range
    normalized = centered / max_dist * ((hi - lo) / 2.0)
    return normalized, center, scale


def denormalize_from_canonical(pc: np.ndarray, center: np.ndarray,
                                scale: float,
                                source_range: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """Reverse canonical normalisation."""
    lo, hi = source_range
    return pc / ((hi - lo) / 2.0) * scale + center


def estimate_normals(pc: np.ndarray,
                     radius: float = 0.1,
                     max_nn: int = 30) -> np.ndarray:
    """Estimate point cloud normals using Open3D."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn))
    return np.asarray(pcd.normals)


def filter_outliers(pc: np.ndarray, nb_neighbors: int = 20,
                    std_ratio: float = 2.0) -> np.ndarray:
    """Remove statistical outliers."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(pcd.points)
