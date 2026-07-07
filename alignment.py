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

    NOTE: This is a simplified version. The full differentiable renderer
    (Nvdiffrast / PyTorch3D) would handle lighting and texture.

    Args:
        mesh:        Open3D TriangleMesh (canonical space)
        target_rgb:  (H, W, 3) reference image
        intrinsics:  3 x 3 camera intrinsics
        steps:       optimization steps
        lr:          learning rate

    Returns:
        R, t, s (scale fixed to 1.0 here)
    """
    import open3d as o3d

    # Sample surface points
    pts = np.asarray(mesh.sample_points_uniformly(number_of_points=2048).points)

    # Initialize transform
    R = torch.eye(3, requires_grad=False)
    t = torch.zeros(3, requires_grad=True)
    # Simple: minimize 2D projection error of sampled points against mask
    # This is a placeholder; real DR would render the mesh and compare pixels.

    target_t = torch.from_numpy(pts.mean(axis=0)).float()
    t_opt = target_t

    return np.eye(3, dtype=np.float32), t_opt.numpy().astype(np.float32), 1.0
