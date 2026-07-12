import numpy as np
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

import sys, os
os.environ['LIDRA_SKIP_INIT'] = 'true'
sys.path.insert(0, '/mnt/sda/johnli/sam3d_repo')

from omegaconf import OmegaConf
from hydra.utils import instantiate

config_path = '/mnt/sda/johnli/SAMScene/pretrained_weights/sam3d/checkpoints/pipeline.yaml'
config = OmegaConf.load(config_path)
config.rendering_engine = 'pytorch3d'
config.compile_model = False
config.workspace_dir = os.path.dirname(config_path)

print('Loading pipeline...')
pipeline = instantiate(config)
print('Pipeline loaded!')

sd = np.load('/mnt/sda/johnli/SAMScene/test/output/scene_data.npz', allow_pickle=True)
img = sd['image']

# Test with object_01 (houseplant) which has smaller bbox
for obj_id in range(7):
    d = np.load(f'/mnt/sda/johnli/SAMScene/test/output/object_{obj_id:02d}.npz', allow_pickle=True)
    mask = d['mask']
    bbox = d['bbox']
    name = str(d['name'])
    
    x1,y1,x2,y2 = bbox
    margin = int(min(x2-x1, y2-y1) * 0.2)
    x1c = max(0, x1-margin); y1c = max(0, y1-margin)
    x2c = min(img.shape[1], x2+margin); y2c = min(img.shape[0], y2+margin)
    crop_img = img[y1c:y2c, x1c:x2c]
    crop_mask = mask[y1c:y2c, x1c:x2c]
    
    mask_alpha = (crop_mask.astype(np.uint8) * 255)[..., None]
    rgba = np.concatenate([crop_img[..., :3], mask_alpha], axis=-1)
    
    print(f'\n[{obj_id}] {name}: crop={crop_img.shape}, mask_sum={crop_mask.sum()}, rgba={rgba.shape}')
    
    try:
        output = pipeline.run(rgba, None, seed=42, stage1_only=True)
        print(f'  Stage1 OK: rotation={output["rotation"].shape}, scale={output["scale"]}')
    except Exception as e:
        print(f'  FAILED: {e}')
        break

print('\nAll stage1 tests done!')
