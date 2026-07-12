import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
import utils3d.numpy

# Monkey-patch missing functions
def _depth_edge(depth, rtol=0.03, mask=None):
    if mask is None:
        mask = np.ones_like(depth, dtype=bool)
    d = np.where(mask, depth, np.nan)
    d_max = maximum_filter(np.nan_to_num(d, nan=-np.inf), size=3)
    d_min = minimum_filter(np.nan_to_num(d, nan=np.inf), size=3)
    return (d_max - d_min) > rtol * (np.abs(depth) + 1e-6)

def _normals_edge(normals, rtol=0.1, mask=None):
    if mask is None:
        mask = np.ones(normals.shape[:2], dtype=bool)
    h, w = normals.shape[:2]
    edge = np.zeros((h, w), dtype=bool)
    for di, dj in [(0,1),(1,0),(1,1),(-1,1)]:
        shifted = np.roll(np.roll(normals, di, axis=0), dj, axis=1)
        dot = np.abs(np.sum(normals * shifted, axis=-1))
        dot = np.clip(dot, 0, 1)
        edge |= (dot < np.cos(rtol))
    edge[~mask] = False
    return edge

utils3d.numpy.depth_edge = _depth_edge
utils3d.numpy.normals_edge = _normals_edge

import sys, os
os.environ['LIDRA_SKIP_INIT'] = 'true'
sys.path.insert(0, '/mnt/sda/johnli/sam3d_repo')

from omegaconf import OmegaConf
from hydra.utils import instantiate

config_path = '/mnt/sda/johnli/SAMScene/pretrained_weights/sam3d/checkpoints/pipeline.yaml'
print(f'Loading: {config_path}')

config = OmegaConf.load(config_path)
config.rendering_engine = 'pytorch3d'
config.compile_model = False
config.workspace_dir = os.path.dirname(config_path)

print('Instantiatiating pipeline...')
pipeline = instantiate(config)
print('SUCCESS: Model loaded!')
