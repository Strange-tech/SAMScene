"""
SAM 3D Wrapper — replaces ObjectGen in the CAST pipeline.

Wraps Meta's SAM 3D Objects model (facebook/sam-3d-objects) to provide
a unified interface matching CAST's expectations: single RGB image + mask
→ textured 3D mesh + 6D pose (rotation, translation, scale).

Paper reference:
  SAM 3D Objects — Meta Superintelligence Labs (Nov 2025)
  https://ai.meta.com/research/sam3d/

Architecture recap:
  - Stage 1: DINOv2 encoder → 1.2B Flow Matching Transformer (MoT)
    → coarse voxel shape + 6D layout (R, t, s)
  - Stage 2: Sparse latent flow matching on active voxels
    → refined geometry + texture → VAE decode → mesh / Gaussian splat

Requirements:
  pip install transformers huggingface_hub
  Model: facebook/sam-3d-objects  (~7 GB, downloaded on first use)

If the model is unavailable (no network / GPU too small), falls back to
a simple shape generator so the rest of the pipeline remains testable.
"""

import warnings
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from typing import Optional, Tuple, Dict

import numpy as np
import torch

# ============================================================================
# 1. SAM 3D model loader
# ============================================================================

class SAM3DWrapper:
    """
    Unified wrapper around Meta SAM 3D Objects for single-image 3D generation.

    Usage:
        sam3d = SAM3DWrapper(device="cuda")
        mesh, R, t, s = sam3d.generate(image, mask)
        # mesh: Open3D TriangleMesh with vertex colors (texture)
        # R, t, s: similarity transform (camera space)
    """

    def __init__(self,
                 model_id: str = "facebook/sam-3d-objects",
                 device: str = "cuda",
                 use_fp16: bool = True,
                 offline: bool = False):
        """
        Args:
            model_id: HuggingFace model ID
            device:   "cuda" or "cpu"
            use_fp16: use half-precision (saves ~3.5 GB VRAM)
            offline:  if True, skip download attempt; use local cache only
        """
        self.model_id = model_id
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None
        self._processor = None
        self._available = False

        if not offline:
            self._init_model()

    def _init_model(self):
        """Attempt to load SAM 3D from HuggingFace."""
        try:
            from transformers import AutoModel, AutoProcessor

            print(f"[SAM3D] Loading {self.model_id} ...")
            dtype = torch.float16 if self.use_fp16 else torch.float32

            self._model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self.device).eval()

            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            self._available = True
            print(f"[SAM3D] Model loaded successfully on {self.device}.")

        except ImportError:
            warnings.warn(
                "[SAM3D] transformers not installed. "
                "Install with: pip install transformers huggingface_hub"
            )
        except Exception as e:
            warnings.warn(
                f"[SAM3D] Could not load model '{self.model_id}': {e}\n"
                "Falling back to stub generator. The pipeline will still run "
                "but generated shapes will be low-quality placeholders."
            )

    @property
    def is_available(self) -> bool:
        return self._available and self._model is not None

    # ------------------------------------------------------------------
    # 2. Main generation interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self,
                 image: np.ndarray,
                 mask: np.ndarray,
                 output_format: str = "mesh",
                 num_inference_steps: int = 50,
                 guidance_scale: float = 3.0,
                 ) -> Tuple["open3d.geometry.TriangleMesh",
                            np.ndarray, np.ndarray, float]:
        """
        Generate a textured 3D mesh + 6D pose from a masked image crop.

        Args:
            image:          (H, W, 3) uint8 RGB crop of the object
            mask:           (H, W) uint8 binary mask (1 = object)
            output_format:  "mesh" or "gaussian" (3DGS .ply)
            num_inference_steps: flow matching steps
            guidance_scale:      CFG scale

        Returns:
            mesh:  Open3D TriangleMesh (with vertex colors if texture available)
            R:     3×3 rotation matrix (camera → object canonical)
            t:     3-vector translation
            s:     uniform scale
        """
        if self.is_available:
            return self._generate_real(image, mask, output_format,
                                       num_inference_steps, guidance_scale)
        else:
            return self._generate_stub(image, mask)

    # ------------------------------------------------------------------
    # 2a. Real SAM 3D inference
    # ------------------------------------------------------------------

    def _generate_real(self, image, mask, output_format,
                       num_inference_steps, guidance_scale):
        """Run the actual SAM 3D model."""
        from PIL import Image as PILImage

        # Prepare inputs
        pil_image = PILImage.fromarray(image)

        # SAM 3D expects the image + optionally a mask
        # The exact API depends on the model's processor; adapt as needed.
        #
        # Typical inference pattern (based on SAM 3D's HuggingFace model card):
        inputs = self._processor(
            images=pil_image,
            masks=PILImage.fromarray(mask * 255),
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run the model
        outputs = self._model(
            **inputs,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type=output_format,
        )

        # Parse outputs
        mesh = self._extract_mesh(outputs)
        R, t, s = self._extract_pose(outputs)

        return mesh, R, t, s

    # ------------------------------------------------------------------
    # 2b. Stub fallback (when SAM 3D is unavailable)
    # ------------------------------------------------------------------

    def _generate_stub(self, image, mask):
        """Generate a simple placeholder shape. Keeps the pipeline testable."""
        import open3d as o3d

        warnings.warn(
            "[SAM3D] Using stub generator — output is a placeholder sphere."
        )

        h, w = image.shape[:2]

        # Rough scale estimate from mask size
        mask_area = mask.sum()
        img_area = h * w
        scale = np.sqrt(mask_area / img_area) * 2.0  # heuristic

        # Rough translation: center of mask → normalized coords
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            cy, cx = ys.mean() / h - 0.5, xs.mean() / w - 0.5
            t = np.array([cx, -cy, 1.5], dtype=np.float32)
        else:
            t = np.array([0.0, 0.0, 1.5], dtype=np.float32)

        R = np.eye(3, dtype=np.float32)

        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        mesh.compute_vertex_normals()
        # Apply a simple gray color
        mesh.paint_uniform_color([0.7, 0.7, 0.7])

        return mesh, R, t, scale

    # ------------------------------------------------------------------
    # 3. Output parsing
    # ------------------------------------------------------------------

    def _extract_mesh(self, outputs) -> "open3d.geometry.TriangleMesh":
        """
        Parse SAM 3D output → Open3D TriangleMesh.

        SAM 3D can output either:
          - A .obj/.glb mesh path/buffer
          - A .ply Gaussian splat
        We convert both to Open3D TriangleMesh for downstream compatibility.
        """
        import open3d as o3d

        # Depending on output_format, the model returns different structures.
        # Adapt based on the actual SAM 3D API.

        if hasattr(outputs, 'mesh') and outputs.mesh is not None:
            # Option A: returns a trimesh / mesh object directly
            # Convert to Open3D
            mesh_data = outputs.mesh
            if hasattr(mesh_data, 'vertices') and hasattr(mesh_data, 'faces'):
                # Assume trimesh-like
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(
                    np.asarray(mesh_data.vertices))
                o3d_mesh.triangles = o3d.utility.Vector3iVector(
                    np.asarray(mesh_data.faces))
                if hasattr(mesh_data, 'visual') and hasattr(mesh_data.visual, 'vertex_colors'):
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                        np.asarray(mesh_data.visual.vertex_colors)[:, :3] / 255.0)
                o3d_mesh.compute_vertex_normals()
                return o3d_mesh

        if hasattr(outputs, 'mesh_path') and outputs.mesh_path:
            # Option B: returns a file path
            path = outputs.mesh_path if isinstance(outputs.mesh_path, str) else outputs.mesh_path[0]
            mesh = o3d.io.read_triangle_mesh(path)
            mesh.compute_vertex_normals()
            return mesh

        if hasattr(outputs, 'vertices') and hasattr(outputs, 'faces'):
            # Option C: raw vertices + faces tensors
            verts = outputs.vertices.cpu().numpy() if torch.is_tensor(outputs.vertices) else np.asarray(outputs.vertices)
            faces = outputs.faces.cpu().numpy() if torch.is_tensor(outputs.faces) else np.asarray(outputs.faces)
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d_mesh.compute_vertex_normals()
            return o3d_mesh

        # Fallback
        warnings.warn("[SAM3D] Could not parse mesh from outputs — using placeholder.")
        return o3d.geometry.TriangleMesh.create_sphere(radius=0.3)

    def _extract_pose(self, outputs) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract (R, t, s) from SAM 3D's 6D layout prediction.

        SAM 3D Stage 1 predicts a 6D layout: rotation (6D continuous repr),
        translation (3D), and uniform scale (1D).
        """
        if hasattr(outputs, 'rotation') and hasattr(outputs, 'translation'):
            # Direct attributes
            R = outputs.rotation
            t = outputs.translation
            s = outputs.scale if hasattr(outputs, 'scale') else 1.0

            if torch.is_tensor(R):
                R = R.cpu().numpy()
            if torch.is_tensor(t):
                t = t.cpu().numpy()
            if torch.is_tensor(s):
                s = s.item()

            # If R is 6D representation, convert to 3×3 matrix
            R = np.asarray(R)
            if R.shape == (6,):
                R = _rotation_6d_to_3x3(R)
            elif R.shape == (3, 3):
                pass  # already a matrix
            else:
                R = R.reshape(3, 3)

            return R.astype(np.float32), np.asarray(t, dtype=np.float32).flatten()[:3], float(s)

        if hasattr(outputs, 'layout') and outputs.layout is not None:
            # 10D layout: 6D rotation + 3D translation + 1D scale
            layout = outputs.layout.cpu().numpy() if torch.is_tensor(outputs.layout) else np.asarray(outputs.layout)
            layout = layout.flatten()
            r6d = layout[:6]
            t = layout[6:9]
            s = layout[9] if len(layout) > 9 else 1.0
            R = _rotation_6d_to_3x3(r6d)
            return R.astype(np.float32), t.astype(np.float32), float(s)

        # Fallback
        warnings.warn("[SAM3D] Could not parse pose from outputs — using identity.")
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0


# ============================================================================
# 4. Helper: 6D rotation → 3×3 matrix
# ============================================================================

def _rotation_6d_to_3x3(r6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D continuous rotation representation (Zhou et al. 2019)
    to a 3×3 rotation matrix via Gram-Schmidt orthogonalization.
    """
    a1 = r6d[:3]
    a2 = r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # 3×3
