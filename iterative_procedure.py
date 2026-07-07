"""
Object generation + alignment procedure — adapted for SAM 3D.

With SAM 3D, the original CAST iterative loop (ObjectGen ↔ AlignGen) is
replaced by a streamlined flow:

  1. SAM 3D one-shot inference → mesh + texture + 6D camera-space pose
  2. PoseAdapter bridges camera space → scene space
  3. Optional ICP refinement against the MoGe scene point cloud

This eliminates the need for:
  - Canonical-space point cloud conditioning (SAM 3D handles it implicitly)
  - Iterative beta-schedule refinement
  - Separate AlignGen (SAM 3D's 6D layout replaces it)

The function signature is kept compatible with the original pipeline.py
so minimal changes are needed there.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

from sam3d_wrapper import SAM3DWrapper
from pose_adapter import PoseAdapter
from utils.point_cloud import fps, normalize_to_canonical


def generate_with_sam3d(
    image: np.ndarray,
    mask: np.ndarray,
    scene_pc: np.ndarray,
    sam3d: SAM3DWrapper,
    pose_adapter: Optional[PoseAdapter] = None,
    verbose: bool = True,
) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray, float]:
    """
    Generate a 3D object mesh + scene-space pose using SAM 3D.

    Args:
        image:         (H, W, 3) uint8 RGB crop of the object
        mask:          (H, W) uint8 binary mask
        scene_pc:      (N, 3) partial point cloud in scene coordinates (from MoGe)
        sam3d:         SAM3DWrapper instance
        pose_adapter:  PoseAdapter instance (None -> skip refinement)
        verbose:       print progress

    Returns:
        mesh:   Open3D TriangleMesh (with vertex colors if texture available)
        R:      3x3 rotation (canonical -> scene)
        t:      3-vector translation
        s:      uniform scale
    """
    if verbose:
        print("  [SAM3D] One-shot generation + pose estimation ...")

    # ---- Step 1: SAM 3D inference ----
    mesh, R_cam, t_cam, s_cam = sam3d.generate(
        image=image,
        mask=mask,
    )

    if verbose:
        n_verts = len(mesh.vertices) if mesh is not None else 0
        print(f"  [SAM3D] Generated mesh: {n_verts} vertices")
        print(f"  [SAM3D] Camera-space pose: s={s_cam:.3f}, "
              f"t=({t_cam[0]:.2f},{t_cam[1]:.2f},{t_cam[2]:.2f})")

    # ---- Step 2: Pose adaptation (camera -> scene coordinate system) ----
    if pose_adapter is not None and len(scene_pc) > 10:
        if verbose:
            print(f"  [PoseAdapter] Bridging camera->scene space "
                  f"(refinement={pose_adapter.refinement}) ...")
        R_scene, t_scene, s_scene = pose_adapter.adapt(
            mesh=mesh,
            sam3d_R=R_cam,
            sam3d_t=t_cam,
            sam3d_s=s_cam,
            scene_point_cloud=scene_pc,
            object_mask=mask,
        )
    else:
        if verbose:
            print("  [PoseAdapter] Skipped (no adapter or insufficient scene PC).")
        R_scene, t_scene, s_scene = R_cam, t_cam, s_cam

    return mesh, R_scene, t_scene, s_scene


# ============================================================================
# Backward-compatible wrapper
# ============================================================================

def iterative_generate(
    image: np.ndarray,
    mask: np.ndarray,
    scene_pc: np.ndarray,
    sam3d: SAM3DWrapper,
    pose_adapter: Optional[PoseAdapter] = None,
    # The following kwargs are kept for backward compatibility but unused
    # with SAM 3D (they were for the old ObjectGen+AlignGen loop).
    max_iterations: int = 1,
    convergence_threshold: float = 0.01,
    diffusion_steps: int = 50,
    pc_num_points: int = 2048,
    use_icp_fallback: bool = True,
    verbose: bool = True,
) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray, float]:
    """
    Backward-compatible entry point. Delegates to generate_with_sam3d().

    Old signature (ObjectGen + AlignGen):
        iterative_generate(image, mask, scene_pc,
                           objectgen, aligngen,
                           max_iterations, convergence_threshold,
                           diffusion_steps, pc_num_points, use_icp_fallback)

    New signature (SAM 3D):
        iterative_generate(image, mask, scene_pc,
                           sam3d, pose_adapter,
                           ...)  # extra kwargs are accepted but ignored

    This allows pipeline.py to call the same function name with minimal changes.
    """
    return generate_with_sam3d(
        image=image,
        mask=mask,
        scene_pc=scene_pc,
        sam3d=sam3d,
        pose_adapter=pose_adapter,
        verbose=verbose,
    )
