"""
Pose adapter — bridges SAM 3D's camera-space 6D layout to CAST's
scene-space coordinate frame, with optional ICP refinement.

Two key responsibilities:

1. Coordinate bridge:
   SAM 3D outputs poses relative to the camera/view. CAST's downstream
   stages (physics correction, scene assembly) expect poses in a unified
   scene coordinate system (derived from MoGe's depth point cloud).

2. ICP refinement:
   SAM 3D's predicted pose may not perfectly align with the observed
   depth point cloud. We can optionally run a lightweight ICP between
   the generated mesh (projected to the pose) and the scene-space partial
   point cloud, refining (R, t) while keeping the scale s fixed.

The adapter supports three refinement levels:
  - "none"     : direct pass-through
  - "umeyama"  : closed-form SVD alignment to scene PC
  - "icp"      : iterative closest point (Open3D, point-to-point)
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

from alignment import umeyama, icp_align


class PoseAdapter:
    """
    Converts SAM 3D camera-space poses into CAST scene-space transforms,
    with optional geometric refinement against observed point clouds.
    """

    def __init__(self,
                 refinement: str = "icp",
                 camera_intrinsics: Optional[np.ndarray] = None,
                 icp_max_distance: float = 0.1,
                 verbose: bool = True):
        """
        Args:
            refinement:        "none" | "umeyama" | "icp"
            camera_intrinsics: 3×3 intrinsics (for possible 2D→3D mapping)
            icp_max_distance:  max correspondence distance for ICP (meters)
            verbose:           print alignment diagnostics
        """
        self.refinement = refinement
        self.camera_intrinsics = camera_intrinsics
        self.icp_max_distance = icp_max_distance
        self.verbose = verbose

    # ------------------------------------------------------------------
    def adapt(self,
              mesh: o3d.geometry.TriangleMesh,
              sam3d_R: np.ndarray,        # 3×3, camera → object canonical
              sam3d_t: np.ndarray,        # 3,    camera-space translation
              sam3d_s: float,             # uniform scale
              scene_point_cloud: np.ndarray,  # (N, 3) from depth → world
              object_mask: Optional[np.ndarray] = None,
              ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Transform SAM 3D pose → CAST scene-space transform.

        Args:
            mesh:              generated mesh (canonical space)
            sam3d_R, sam3d_t, sam3d_s: SAM 3D output pose
            scene_point_cloud: (N, 3) partial point cloud from MoGe depth
            object_mask:       optional (H,W) mask for filtering the PC

        Returns:
            R_scene:  3×3 rotation (canonical → scene)
            t_scene:  3 vector translation
            s_scene:  uniform scale
        """
        # Step 1: Normalize the scene point cloud
        if len(scene_point_cloud) == 0:
            if self.verbose:
                print("[PoseAdapter] Empty scene PC — passing SAM 3D pose through.")
            return sam3d_R, sam3d_t, sam3d_s

        # Filter outliers (optional)
        scene_pc = self._filter_pc(scene_point_cloud)

        # Step 2: Sample mesh surface in canonical space
        mesh_pts = np.asarray(
            mesh.sample_points_uniformly(number_of_points=min(2048, len(scene_pc))).points
        )

        # Step 3: Apply SAM 3D's transform to place mesh in camera space
        mesh_in_camera = sam3d_s * (mesh_pts @ sam3d_R.T) + sam3d_t

        # Step 4: Refine alignment to scene PC
        if self.refinement == "none":
            R_out, t_out, s_out = sam3d_R, sam3d_t, sam3d_s

        elif self.refinement == "umeyama":
            # Closed-form SVD alignment
            if self.verbose:
                print("[PoseAdapter] Running Umeyama refinement ...")
            # Ensure same point count
            n_pts = min(len(mesh_in_camera), len(scene_pc))
            src = mesh_in_camera[:n_pts]
            dst = scene_pc[:n_pts]
            R_ref, t_ref, s_ref = umeyama(dst, src, with_scale=False)
            # Compose with SAM 3D original
            R_out = R_ref @ sam3d_R
            t_out = R_ref @ sam3d_t + t_ref
            s_out = sam3d_s  # keep original scale
            if self.verbose:
                delta = np.linalg.norm(t_ref)
                print(f"  Umeyama delta translation: {delta:.4f} m")

        elif self.refinement == "icp":
            # Iterative closest point
            if self.verbose:
                print("[PoseAdapter] Running ICP refinement ...")
            R_icp, t_icp, _ = icp_align(
                mesh_in_camera, scene_pc, normalize=True
            )
            # Compose
            R_out = R_icp @ sam3d_R
            t_out = R_icp @ sam3d_t + t_icp
            s_out = sam3d_s
            if self.verbose:
                delta = np.linalg.norm(t_icp)
                print(f"  ICP delta translation: {delta:.4f} m")

        else:
            raise ValueError(f"Unknown refinement mode: {self.refinement}")

        return R_out, t_out, s_out

    # ------------------------------------------------------------------
    @staticmethod
    def compute_initial_pose_from_pointcloud(
        scene_pc: np.ndarray,
        canonical_pc: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        When SAM 3D is unavailable, estimate an initial pose purely from
        the scene point cloud (heuristic: center-of-mass + bounding box scale).

        This is useful as a fallback for the rest of the CAST pipeline.
        """
        if scene_pc is None or len(scene_pc) == 0:
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0

        center = scene_pc.mean(axis=0)
        max_extent = np.linalg.norm(scene_pc - center, axis=1).max()
        scale = max_extent if max_extent > 0.01 else 1.0

        return np.eye(3, dtype=np.float32), center.astype(np.float32), float(scale)

    # ------------------------------------------------------------------
    @staticmethod
    def _filter_pc(pc: np.ndarray,
                   std_ratio: float = 2.0,
                   nb_neighbors: int = 20) -> np.ndarray:
        """Remove statistical outliers from a point cloud."""
        if len(pc) < nb_neighbors:
            return pc
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            return np.asarray(pcd.points)
        except Exception:
            return pc


# ============================================================================
# 6D rotation representation (same as physics_correction.py, exposed here)
# ============================================================================

def rotation_3x3_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix → 6D continuous representation."""
    return np.concatenate([R[:, 0], R[:, 1]])  # first two columns


def rotation_6d_to_3x3(r6d: np.ndarray) -> np.ndarray:
    """Convert 6D representation → 3×3 rotation matrix."""
    a1 = r6d[:3]
    a2 = r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)
