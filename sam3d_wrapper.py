'''
SAM 3D Wrapper — directly uses official Inference + make_scene from notebook/inference.py
'''
import os, sys, warnings
import numpy as np
import torch
from scipy.ndimage import maximum_filter, minimum_filter
import utils3d.numpy as _un

# ---- Monkey-patch ALL missing utils3d functions ----
def _depth_edge(depth, rtol=0.03, mask=None):
    if mask is None: mask = np.ones_like(depth, dtype=bool)
    d = np.where(mask, depth, np.nan)
    d_max = maximum_filter(np.nan_to_num(d, nan=-np.inf), size=3)
    d_min = minimum_filter(np.nan_to_num(d, nan=np.inf), size=3)
    return (d_max - d_min) > rtol * (np.abs(depth) + 1e-6)

def _normals_edge(normals, rtol=0.1, mask=None):
    if mask is None: mask = np.ones(normals.shape[:2], dtype=bool)
    h, w = normals.shape[:2]; edge = np.zeros((h, w), dtype=bool)
    for di, dj in [(0,1),(1,0),(1,1),(-1,1)]:
        shifted = np.roll(np.roll(normals, di, axis=0), dj, axis=1)
        dot = np.abs(np.sum(normals * shifted, axis=-1))
        dot = np.clip(dot, 0, 1); edge |= (dot < np.cos(rtol))
    edge[~mask] = False
    return edge

_un.depth_edge = _depth_edge
_un.normals_edge = _normals_edge
_un.points_to_normals = lambda *a, **kw: np.zeros((1,1,3))
_un.image_uv = lambda *a, **kw: (np.zeros((1,1,2)), np.zeros((1,1)))
_un.image_mesh = lambda *a, **kw: np.zeros((0,3))

os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
if '/mnt/sda/johnli/sam3d_repo' not in sys.path:
    sys.path.insert(0, '/mnt/sda/johnli/sam3d_repo')
if '/mnt/sda/johnli/sam3d_repo/notebook' not in sys.path:
    sys.path.insert(0, '/mnt/sda/johnli/sam3d_repo/notebook')

# ---- Directly import official Inference and make_scene ----
from inference import Inference, make_scene


class SAM3DWrapper:
    def __init__(self, device='cuda', config_path=None):
        if config_path is None:
            config_path = '/mnt/sda/johnli/SAMScene/pretrained_weights/sam3d/checkpoints/pipeline.yaml'
        self._inference = Inference(config_path, compile=False)
        self._available = True

    @property
    def is_available(self):
        return self._available

    def generate_scene_ply(self, image, masks, output_path, seed=42):
        '''
        Official demo pipeline:
          outputs = [inference(image, mask) for mask in masks]
          scene_gs = make_scene(*outputs)
          scene_gs.save_ply(output_path)
        '''
        print(f'[SAM3D] Generating {len(masks)} objects (official Inference API) ...')
        outputs = []
        for i, mask in enumerate(masks):
            print(f'  [{i}] running inference, mask_nonzero={mask.sum()} ...')
            output = self._inference(image, mask, seed=seed)
            outputs.append(output)

        print(f'[SAM3D] Combining into scene (make_scene) ...')
        scene_gs = make_scene(*outputs)

        print(f'[SAM3D] Saving to {output_path} ...')
        scene_gs.save_ply(output_path)

        # Also extract individual mesh info for downstream use
        meshes, poses = [], []
        for i, output in enumerate(outputs):
            mesh = self._extract_mesh(output)
            R, t, s = self._extract_pose(output)
            meshes.append(mesh)
            poses.append({'R': R, 't': t, 's': s})

        return output_path, meshes, poses

    def _extract_mesh(self, output):
        import open3d as o3d
        if 'mesh' in output and output['mesh']:
            m = output['mesh'][0]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices.cpu()))
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(m.faces.cpu()))
            mesh.compute_vertex_normals()
            return mesh
        return o3d.geometry.TriangleMesh.create_sphere(radius=0.3)

    def _extract_pose(self, output):
        from pytorch3d.transforms import quaternion_to_matrix
        rot = output.get('rotation')
        if rot is not None:
            if torch.is_tensor(rot): rot = rot.cpu().numpy()
            rot = np.asarray(rot).reshape(-1)
            if rot.shape == (4,):
                R = quaternion_to_matrix(torch.from_numpy(rot).unsqueeze(0)).squeeze(0).numpy()
            else:
                R = np.eye(3)
        else:
            R = np.eye(3)

        t = output.get('translation')
        if t is not None:
            if torch.is_tensor(t): t = t.cpu().numpy()
            t = np.asarray(t, dtype=np.float32).flatten()[:3]
        else:
            t = np.zeros(3, dtype=np.float32)

        s = output.get('scale')
        if s is not None:
            if torch.is_tensor(s): s = s.cpu().flatten()[0].item()
            s = float(np.asarray(s).flatten()[0])
        else:
            s = 1.0

        return R.astype(np.float32), t.astype(np.float32), float(s)
