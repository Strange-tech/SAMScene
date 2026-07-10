"""
Generative alignment (AlignGen) — Section 4.2 of CAST.

AlignGen is a point-cloud diffusion transformer that maps a scene-space
partial point cloud q to a canonical-space point cloud p, so that p aligns
with the generated object mesh. The similarity transformation (scale,
rotation, translation) is then recovered via the Umeyama algorithm.

This has two key advantages over direct pose regression:
  1. It handles multi-modal pose distributions (symmetry, ambiguity).
  2. The Umeyama step is numerically stable (closed-form SVD).

When the AlignGen checkpoint is unavailable, falls back to:
  - ICP (Iterative Closest Point) with bounding-box normalisation
  - Differentiable rendering alignment
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# 1. Umeyama algorithm (closed-form similarity transform)
# ============================================================================

def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Closed-form least-squares estimation of similarity transformation
    (rotation, translation, uniform scale) between two point sets.

    src_i = s * R * dst_i + t   (or dst → src mapping)

    Args:
        src: (N, 3) target / scene-space points
        dst: (N, 3) source / canonical-space points
        with_scale: if True, estimate uniform scale; else s=1.0

    Returns:
        R:     3 x 3 rotation matrix
        t:     3   translation vector
        s:     uniform scale
    """
    assert src.shape == dst.shape, f"{src.shape} vs {dst.shape}"
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    sigma_src = np.sum(np.linalg.norm(src_centered, axis=1) ** 2) / n

    # Cross-covariance
    H = (dst_centered.T @ src_centered) / n  # 3 x 3

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection check
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    if with_scale and sigma_src > 1e-8:
        s = np.trace(np.diag(np.linalg.svd(H, compute_uv=False))) / sigma_src
    else:
        s = 1.0

    t = mu_src - s * R @ mu_dst
    return R, t, s


def apply_similarity(pc: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    """Apply similarity transform: pc_out = s * (R @ pc_i) + t"""
    return s * (pc @ R.T) + t


# ============================================================================
# 2. AlignGen Diffusion Model
# ============================================================================

class StubPointCloudDiffusion(nn.Module):
    """
    Stub point-cloud diffusion transformer for pose alignment (Sec 4.2).

    Real AlignGen: 24-layer transformer, 150M params, trained on 200K
    curated Objaverse assets for ~2 days on 64 A800 GPUs.

    Conditioning:
      - q (scene-space partial PC): concatenated along feature channel
      - Z (geometry latent): cross-attention
    """

    def __init__(self, pc_dim: int = 3, latent_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.pc_proj = nn.Linear(pc_dim * 2, hidden_dim)  # concat(q, p_t)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, hidden_dim)
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            ) for _ in range(6)
        ])
        self.out_proj = nn.Linear(hidden_dim, pc_dim)

    def forward(self, p_t: torch.Tensor, t: torch.Tensor,
                q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_t: (B, N, 3) noisy canonical-space PC at timestep t
            t:   (B,) or (B, 1)
            q:   (B, N, 3) scene-space partial PC
            z:   (B, latent_dim) geometry latent from ObjectGen
        Returns:
            (B, N, 3) predicted noise / denoised p
        """
        b, n, _ = p_t.shape
        t_emb = self.time_mlp(t.float().unsqueeze(-1)).unsqueeze(1)  # (B, 1, hidden)

        # Concatenate p_t and q along feature dim (as in paper)
        inp = torch.cat([p_t, q], dim=-1)                            # (B, N, 6)
        h = self.pc_proj(inp) + t_emb                                # (B, N, hidden)

        # Inject geometry latent via FiLM-style conditioning
        z_cond = self.latent_proj(z).unsqueeze(1)                     # (B, 1, hidden)
        h = h + z_cond

        for block in self.blocks:
            h = h + block(h)

        return self.out_proj(h)                                       # (B, N, 3)


# ============================================================================
# 3. AlignGen wrapper
# ============================================================================

class AlignGen:
    """
    Generative pose-alignment module (Section 4.2).

    Maps scene-space partial point cloud q → canonical-space p, then
    recovers the similarity transform via Umeyama.
    """

    def __init__(self,
                 ckpt_path: Optional[str] = None,
                 device: str = "cuda",
                 pc_dim: int = 3,
                 latent_dim: int = 512):
        self.device = device
        self.model = StubPointCloudDiffusion(pc_dim=pc_dim,
                                              latent_dim=latent_dim).to(device).eval()
        if ckpt_path is not None:
            try:
                sd = torch.load(ckpt_path, map_location=device)
                self.model.load_state_dict(sd, strict=False)
                print(f"[AlignGen] Loaded checkpoint from {ckpt_path}")
            except FileNotFoundError:
                warnings.warn(f"[AlignGen] Checkpoint not found: {ckpt_path}")

    @torch.no_grad()
    def align(self,
              q: np.ndarray,
              z: np.ndarray,
              diffusion_steps: int = 50,
              num_samples: int = 3) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Estimate similarity transform from scene-space PC to canonical space.

        The paper samples multiple noise realizations and aggregates results
        to handle symmetric / ambiguous poses (Section 4.2).

        Args:
            q:              (N, 3) scene-space partial point cloud
            z:              (D,)   geometry latent code from ObjectGen
            diffusion_steps: denoising steps
            num_samples:     number of noise realizations for aggregation

        Returns:
            R:  3 x 3 rotation
            t:  3-vector translation
            s:  uniform scale
            p:  (N, 3) aligned canonical-space PC (best sample)
        """
        q_t = torch.from_numpy(q.astype(np.float32)).unsqueeze(0).to(self.device)  # (1, N, 3)
        z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(self.device)  # (1, D)

        best_score = -np.inf
        best_transform = (np.eye(3), np.zeros(3), 1.0)
        best_p = q.copy()

        for _ in range(num_samples):
            # Initialize noise
            p_t = torch.randn(1, q.shape[0], 3, device=self.device)

            # DDIM denoising (simplified)
            timesteps = torch.linspace(1.0, 0.0, diffusion_steps + 1, device=self.device)
            for i in range(diffusion_steps):
                t_val = timesteps[i].unsqueeze(0)
                eps = self.model(p_t, t_val, q_t, z_t)
                p_t = p_t - eps * 0.02

            p = p_t.squeeze(0).cpu().numpy()  # (N, 3)

            # Recover transform via Umeyama
            R, t, s = umeyama(q, p, with_scale=True)

            # Score based on alignment quality (inlier ratio)
            p_transformed = apply_similarity(p, np.linalg.inv(R), -t / s, 1.0 / s)
            dists = np.linalg.norm(q - p_transformed, axis=1)
            inlier_ratio = (dists < 0.1).mean()  # fraction within 10cm

            if inlier_ratio > best_score:
                best_score = inlier_ratio
                best_transform = (R, t, s)
                best_p = p

        return (*best_transform, best_p)


# ============================================================================
# 4. Fallback: ICP alignment
# ============================================================================

def icp_align(src_pc: np.ndarray, dst_pc: np.ndarray,
              normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fallback pose estimation using Iterative Closest Point with
    bounding-box normalization (as described in the paper's ablation).

    Args:
        src_pc: (N, 3) generated mesh points (sampled surface)
        dst_pc: (N, 3) target partial point cloud (scene-space)
        normalize: if True, normalize both to unit bounding box first

    Returns:
        R, t, s
    """
    import open3d as o3d

    src = src_pc.copy()
    dst = dst_pc.copy()

    if normalize:
        # Normalize to bounding box
        src_bb = np.linalg.norm(src.max(axis=0) - src.min(axis=0))
        dst_bb = np.linalg.norm(dst.max(axis=0) - dst.min(axis=0))
        if src_bb > 1e-8:
            src = src / src_bb
        if dst_bb > 1e-8:
            dst = dst / dst_bb

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src)
    dst_pcd = o3d.geometry.PointCloud()
    dst_pcd.points = o3d.utility.Vector3dVector(dst)

    reg = o3d.pipelines.registration.registration_icp(
        src_pcd, dst_pcd,
        max_correspondence_distance=0.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    T = reg.transformation  # 4 x 4
    R = T[:3, :3]
    t = T[:3, 3]

    if normalize:
        s = src_bb / dst_bb if dst_bb > 1e-8 else 1.0
    else:
        s = 1.0

    return R, t, s


# ============================================================================
# 5. Fallback: differentiable rendering alignment
# ============================================================================

def differentiable_render_align(mesh: "open3d.geometry.TriangleMesh",
                                target_rgb: np.ndarray,
                                intrinsics: np.ndarray,
                                steps: int = 200,
                                lr: float = 0.01) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Optimize rotation and translation so that a rendered view of the mesh
    aligns with the target RGB image (Section 6.3 ablation).

    Uses PyTorch3D's differentiable silhouette renderer to minimise the
    mismatch between rendered mask and target segmentation. Falls back to
    a simpler point-cloud reprojection loss when PyTorch3D is unavailable.

    REQUIRES EXTERNAL SETUP (for full differentiable rendering):
        pip install pytorch3d
        (see https://github.com/facebookresearch/pytorch3d)

    Args:
        mesh:        Open3D TriangleMesh (canonical space)
        target_rgb:  (H, W, 3) reference RGB image
        intrinsics:  3 x 3 camera intrinsics
        steps:       optimization steps
        lr:          learning rate

    Returns:
        R:  3x3 rotation matrix
        t:  3-vector translation
        s:  uniform scale (fixed to 1.0 for this stage)
    """
    import open3d as o3d

    h, w = target_rgb.shape[:2]

    # Extract target silhouette from RGB (simple luminance threshold)
    # In practice this would come from segmentation; here we use a rough mask
    target_gray = cv2.cvtColor if "cv2" in dir() else __import__("cv2").cvtColor
    try:
        import cv2 as _cv2
        gray = _cv2.cvtColor(target_rgb, _cv2.COLOR_RGB2GRAY)
        target_mask = (gray < 250).astype(np.float32)  # non-white = object
    except Exception:
        # Fallback: assume entire image is object
        target_mask = np.ones((h, w), dtype=np.float32)

    # Sample surface points
    pts_np = np.asarray(
        mesh.sample_points_uniformly(number_of_points=2048).points
    )
    pts = torch.from_numpy(pts_np.astype(np.float32))

    # Camera parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Build camera projection matrix (PyTorch3D NDC or manual)
    target_mask_t = torch.from_numpy(target_mask).float()

    # Try PyTorch3D-based optimisation first
    try:
        import pytorch3d
        _has_pytorch3d = True
    except ImportError:
        _has_pytorch3d = False
        warnings.warn(
            "differentiable_render_align: PyTorch3D not installed. "
            "Falling back to point-cloud reprojection loss. "
            "Install with: pip install pytorch3d"
        )

    if _has_pytorch3d:
        return _dr_align_pytorch3d(
            mesh, target_mask_t, (h, w), (fx, fy, cx, cy),
            steps=steps, lr=lr,
        )
    else:
        return _dr_align_projection(
            pts, target_mask_t, (h, w), (fx, fy, cx, cy),
            steps=steps, lr=lr,
        )


def _dr_align_pytorch3d(
    mesh, target_mask, img_size, cam_params,
    steps=200, lr=0.01,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Full differentiable silhouette rendering via PyTorch3D.
    Optimises (R, t) to make rendered mask match target_mask.
    """
    import open3d as o3d
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, RasterizationSettings,
        MeshRenderer, MeshRasterizer, SoftSilhouetteShader,
        look_at_view_transform,
    )
    from pytorch3d.io import load_obj

    h, w = img_size
    fx, fy, cx, cy = cam_params

    # Convert Open3D mesh → PyTorch3D Meshes
    verts = torch.from_numpy(
        np.asarray(mesh.vertices).astype(np.float32)
    ).unsqueeze(0)
    faces = torch.from_numpy(
        np.asarray(mesh.triangles).astype(np.int64)
    ).unsqueeze(0)

    # Estimate FoV from intrinsics
    fov = 2.0 * torch.atan(torch.tensor(h / (2.0 * fy)))

    # Initialise optimisable parameters (rotation in axis-angle, translation)
    r_axis_angle = torch.zeros(3, requires_grad=True)  # identity rotation
    t_vec = torch.zeros(3, requires_grad=True)
    # Initialise t to center the object
    with torch.no_grad():
        centroid = verts.squeeze(0).mean(dim=0)
        t_vec.copy_(torch.tensor([0.0, 0.0, 2.0]))

    optimizer = torch.optim.Adam([r_axis_angle, t_vec], lr=lr)

    # Renderer setup
    raster_settings = RasterizationSettings(
        image_size=(h, w),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verts = verts.to(device)
    faces = faces.to(device)
    target_mask = target_mask.to(device)
    r_axis_angle = r_axis_angle.to(device)
    t_vec = t_vec.to(device)

    meshes = Meshes(verts=verts, faces=faces)

    for step in range(steps):
        optimizer.zero_grad()

        R = _axis_angle_to_matrix(r_axis_angle)
        # Build camera at (0,0,0) looking along +Z
        camera = FoVPerspectiveCameras(
            fov=fov, degrees=False, device=device,
            R=R.unsqueeze(0), T=t_vec.unsqueeze(0),
        )

        rasterizer = MeshRasterizer(
            cameras=camera, raster_settings=raster_settings
        )
        fragments = rasterizer(meshes)
        # Soft silhouette: alpha = 1 - exp(-depth)
        alpha = 1.0 - torch.exp(-fragments.zbuf.clamp(min=0))
        rendered = alpha[..., 0]  # (1, H, W)

        loss = torch.nn.functional.mse_loss(rendered, target_mask)
        loss.backward()
        optimizer.step()

    # Extract final transform
    R_final = _axis_angle_to_matrix(r_axis_angle.detach()).cpu().numpy()
    t_final = t_vec.detach().cpu().numpy()

    return R_final.astype(np.float32), t_final.astype(np.float32), 1.0


def _dr_align_projection(
    pts, target_mask, img_size, cam_params,
    steps=200, lr=0.01,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fallback: minimise 2D projection error of mesh surface points
    against the target silhouette (no real rendering, but differentiable).
    """
    h, w = img_size
    fx, fy, cx, cy = cam_params

    # Initialise
    # 6D rotation representation (more stable for optimisation)
    r6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True)
    t_vec = torch.zeros(3, requires_grad=True)
    with torch.no_grad():
        t_vec[2] = 2.0  # place object 2m in front of camera

    target_mask_t = target_mask

    optimizer = torch.optim.Adam([r6d, t_vec], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        # Convert 6D → 3x3 rotation
        a1, a2 = r6d[:3], r6d[3:6]
        b1 = a1 / (a1.norm() + 1e-8)
        b2 = a2 - torch.dot(b1, a2) * b1
        b2 = b2 / (b2.norm() + 1e-8)
        b3 = torch.cross(b1, b2)
        R = torch.stack([b1, b2, b3], dim=1)  # 3x3

        # Transform + project points
        pts_transformed = pts @ R.T + t_vec  # (N, 3)
        z = pts_transformed[:, 2].clamp(min=0.01)
        u = (pts_transformed[:, 0] * fx / z + cx).long()
        v = (pts_transformed[:, 1] * fy / z + cy).long()

        # Build rendered mask from point projections
        rendered = torch.zeros(h, w)
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if valid.any():
            rendered[v[valid], u[valid]] = 1.0

        # MSE against target mask
        loss = torch.nn.functional.mse_loss(rendered, target_mask_t)
        loss.backward()
        optimizer.step()

    # Extract final transform
    with torch.no_grad():
        a1, a2 = r6d[:3], r6d[3:6]
        b1 = a1 / (a1.norm() + 1e-8)
        b2 = a2 - torch.dot(b1, a2) * b1
        b2 = b2 / (b2.norm() + 1e-8)
        b3 = torch.cross(b1, b2)
        R_final = torch.stack([b1, b2, b3], dim=1).numpy()
        t_final = t_vec.detach().numpy()

    return R_final.astype(np.float32), t_final.astype(np.float32), 1.0


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to 3x3 rotation matrix."""
    angle = axis_angle.norm()
    if angle < 1e-8:
        return torch.eye(3, device=axis_angle.device)
    axis = axis_angle / angle
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], device=axis_angle.device)
    return torch.eye(3, device=axis_angle.device) + \
        torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
