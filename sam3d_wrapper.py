'''
SAM 3D Wrapper — uses hydra.utils.instantiate.
Multi-object: pass FULL image + per-object mask (no external crop).
'''
import os, sys, warnings
import numpy as np
import torch
from scipy.ndimage import maximum_filter, minimum_filter
import utils3d.numpy as _un

def _depth_edge(depth, rtol=0.03, mask=None):
    if mask is None: mask = np.ones_like(depth, dtype=bool)
    d = np.where(mask, depth, np.nan)
    d_max = maximum_filter(np.nan_to_num(d, nan=-np.inf), size=3)
    d_min = minimum_filter(np.nan_to_num(d, nan=np.inf), size=3)
    return (d_max - d_min) > rtol * (np.abs(depth) + 1e-6)

def _normals_edge(normals, rtol=0.1, mask=None):
    if mask is None: mask = np.ones(normals.shape[:2], dtype=bool)
    h, w = normals.shape[:2]
    edge = np.zeros((h, w), dtype=bool)
    for di, dj in [(0,1),(1,0),(1,1),(-1,1)]:
        shifted = np.roll(np.roll(normals, di, axis=0), dj, axis=1)
        dot = np.abs(np.sum(normals * shifted, axis=-1))
        dot = np.clip(dot, 0, 1)
        edge |= (dot < np.cos(rtol))
    edge[~mask] = False
    return edge

_un.depth_edge = _depth_edge
_un.normals_edge = _normals_edge

os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
_sam3d_repo = '/mnt/sda/johnli/sam3d_repo'
if _sam3d_repo not in sys.path:
    sys.path.insert(0, _sam3d_repo)

from omegaconf import OmegaConf
from hydra.utils import instantiate


class SAM3DWrapper:
    def __init__(self, model_id='facebook/sam-3d-objects', device='cuda',
                 use_fp16=True, offline=False, config_path=None):
        self.device = device
        self._pipeline = None
        self._available = False
        if config_path is None:
            config_path = '/mnt/sda/johnli/SAMScene/pretrained_weights/sam3d/checkpoints/pipeline.yaml'
        self._init_model(config_path)

    def _init_model(self, config_path):
        try:
            print(f'[SAM3D] Loading from {config_path} ...')
            config = OmegaConf.load(config_path)
            config.rendering_engine = 'pytorch3d'
            config.compile_model = False
            config.workspace_dir = os.path.dirname(config_path)
            self._pipeline = instantiate(config)
            self._available = True
            print(f'[SAM3D] Model loaded on {self.device}.')
        except Exception as e:
            warnings.warn(f'[SAM3D] Load failed: {e}')

    @property
    def is_available(self):
        return self._available

    @torch.no_grad()
    def generate(self, image, mask,
                 output_format='mesh',
                 num_inference_steps=50,
                 guidance_scale=3.0):
        '''
        Generate mesh + pose from FULL image + per-object mask.
        image: (H, W, 3) uint8 RGB — FULL image, NOT cropped
        mask:  (H, W) uint8 — per-object binary mask
        Returns: (mesh, R_3x3, t_3, scale_float)
        '''
        if not self.is_available:
            return self._generate_stub(image, mask)

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Embed mask in alpha channel (same as Inference.merge_mask_to_rgba)
        mask_u8 = (mask.astype(np.uint8) * 255)[..., None]
        rgba = np.concatenate([image[..., :3], mask_u8], axis=-1)

        print(f'[SAM3D] Running on full image {rgba.shape} ...')
        output = self._pipeline.run(
            rgba, None, seed=42,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
        )

        mesh = self._extract_mesh(output)
        R, t, s = self._extract_pose(output)
        return mesh, R, t, s

    def _generate_stub(self, image, mask):
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        return mesh, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0

    def _extract_mesh(self, outputs):
        import open3d as o3d
        if 'mesh' in outputs and outputs['mesh']:
            m = outputs['mesh'][0]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices.cpu()))
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(m.faces.cpu()))
            mesh.compute_vertex_normals()
            return mesh
        return o3d.geometry.TriangleMesh.create_sphere(radius=0.3)

    def _extract_pose(self, outputs):
        rot = outputs.get('rotation')
        if rot is not None:
            if torch.is_tensor(rot): rot = rot.cpu().numpy()
            rot = np.asarray(rot).reshape(-1)
            if rot.shape == (4,):  # quaternion → 3x3
                from pytorch3d.transforms import quaternion_to_matrix
                R = quaternion_to_matrix(torch.from_numpy(rot).unsqueeze(0)).squeeze(0).numpy()
            elif rot.shape == (9,):
                R = rot.reshape(3, 3)
            else:
                R = np.eye(3)
        else:
            R = np.eye(3)

        t = outputs.get('translation')
        if t is not None:
            if torch.is_tensor(t): t = t.cpu().numpy()
            t = np.asarray(t, dtype=np.float32).flatten()[:3]
        else:
            t = np.zeros(3, dtype=np.float32)

        s = outputs.get('scale')
        if s is not None:
            if torch.is_tensor(s): s = s.cpu().flatten()[0].item()
            s = float(np.asarray(s).flatten()[0])
        else:
            s = 1.0

        return R.astype(np.float32), t.astype(np.float32), float(s)
