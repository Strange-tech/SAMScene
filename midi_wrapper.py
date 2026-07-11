"""
MIDI Wrapper — 替代 SAM 3D 作为 test_03 的 3D 生成后端.

将 MIDI (CVPR 2025) 的多实例扩散模型封装为与 SAM3DWrapper 兼容的接口，
从单张场景图像 + 分割 mask 同时生成所有物体的 3D mesh。

与 SAM 3D 的关键区别:
  - SAM 3D: 逐物体生成 (image crop -> mesh)，需要 PoseAdapter 桥接坐标
  - MIDI:     所有物体同时生成，已在统一的场景坐标系中，无需 PoseAdapter

Usage:
    wrapper = MIDIWrapper(device="cuda")
    meshes, transforms = wrapper.generate_from_scene(image, masks)
    # meshes[i]: Open3D TriangleMesh (已在场景坐标系中)
"""

import os
import sys
import types
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Patch: 全局强制 IPv4 — 服务器 IPv6 不通，所有 HTTP 库 (requests/httpx/urllib3)
# 默认优先用 IPv6，导致连接 huggingface.co 超时
# ---------------------------------------------------------------------------
import socket as _socket
_orig_getaddrinfo = _socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, _socket.AF_INET, type, proto, flags)
_socket.getaddrinfo = _ipv4_getaddrinfo

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIDI_REPO_ID = "VAST-AI/MIDI-3D"
MIDI_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pretrained_weights", "midi_3d"
)
# midi_3d 源码路径 (假设克隆到 SAMScene 同级目录)
MIDI_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "MIDI-3D"
)

_midi_pipeline = None


# ---------------------------------------------------------------------------
# torch_cluster shim — 纯 PyTorch FPS 替代
# ---------------------------------------------------------------------------

def _fps_pytorch(
    points: torch.Tensor,
    batch: torch.Tensor,
    ratio: float = 1.0,
    random_start: bool = False,
) -> torch.Tensor:
    """
    纯 PyTorch Farthest Point Sampling，接口与 torch_cluster.fps 兼容.

    Args:
        points:  (N, 3) float tensor — 点云坐标
        batch:   (N,)  long tensor   — 每个点的 batch index
        ratio:   float              — 采样比例 (0~1)
        random_start: bool          — 是否随机选择起始点

    Returns:
        (M,) long tensor — 全局索引 (跨 batch)
    """
    device = points.device
    N = points.shape[0]

    if ratio >= 1.0 or N == 0:
        return torch.arange(N, device=device)

    unique_batches = torch.unique(batch)
    all_indices: List[torch.Tensor] = []

    for b in unique_batches:
        mask = (batch == b)
        batch_points = points[mask]  # (n_b, 3)
        n_b = batch_points.shape[0]
        m_b = max(int(n_b * ratio), 1)

        if m_b >= n_b:
            local_idx = torch.arange(n_b, device=device)
        else:
            local_idx = torch.zeros(m_b, dtype=torch.long, device=device)
            dists = torch.full((n_b,), float("inf"), device=device)

            start = 0
            if random_start:
                start = torch.randint(0, n_b, (1,), device=device).item()
            local_idx[0] = start
            farthest = batch_points[start]

            for i in range(1, m_b):
                # 计算当前 farthest 点到所有点的距离
                d = ((batch_points - farthest) ** 2).sum(dim=1)
                dists = torch.minimum(dists, d)
                local_idx[i] = torch.argmax(dists)
                farthest = batch_points[local_idx[i]]

        # 局部索引 → 全局索引
        global_positions = torch.where(mask)[0]
        all_indices.append(global_positions[local_idx])

    if len(all_indices) == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    return torch.cat(all_indices)


def _install_torch_cluster_shim():
    """将纯 PyTorch FPS 实现注入 sys.modules，替代 torch_cluster 包."""
    if "torch_cluster" in sys.modules:
        return  # 已存在（可能是真正的 torch_cluster）

    shim = types.ModuleType("torch_cluster")
    shim.fps = _fps_pytorch
    sys.modules["torch_cluster"] = shim


# ---------------------------------------------------------------------------
# 模型下载 & 加载
# ---------------------------------------------------------------------------

def _ensure_midi_weights():
    """下载 MIDI 模型权重到本地目录 (带进度输出)."""
    if os.path.exists(os.path.join(MIDI_LOCAL_DIR, "model_index.json")):
        print(f"[MIDI] 模型权重已存在: {MIDI_LOCAL_DIR}")
        return

    from huggingface_hub import snapshot_download, list_repo_files

    print(f"[MIDI] 连接 HuggingFace 获取文件列表 ...")
    try:
        all_files = list_repo_files(MIDI_REPO_ID)
        print(f"[MIDI] 仓库共 {len(all_files)} 个文件，开始下载 ...")
        print(f"       目标目录: {MIDI_LOCAL_DIR}")
        print(f"       (模型约 5-10 GB, 预计需要数分钟至数十分钟)")
    except Exception:
        print(f"[MIDI] 正在下载 {MIDI_REPO_ID} -> {MIDI_LOCAL_DIR}")

    try:
        from tqdm import tqdm
        snapshot_download(
            repo_id=MIDI_REPO_ID,
            local_dir=MIDI_LOCAL_DIR,
            tqdm_class=tqdm,
            resume_download=True,
            max_workers=2,
        )
    except ImportError:
        snapshot_download(
            repo_id=MIDI_REPO_ID,
            local_dir=MIDI_LOCAL_DIR,
            resume_download=True,
            max_workers=2,
        )

    print("[MIDI] 下载完成.")


def _get_midi_pipeline():
    """懒加载 MIDI pipeline (全局单例)."""
    global _midi_pipeline
    if _midi_pipeline is not None:
        return _midi_pipeline

    try:
        _ensure_midi_weights()

        # 在导入 MIDI 之前注入 torch_cluster shim
        _install_torch_cluster_shim()

        # 将 midi_3d 源码目录加入 Python path
        midi_src = os.path.realpath(MIDI_SOURCE_DIR)
        if midi_src not in sys.path:
            sys.path.insert(0, midi_src)

        from midi.pipelines.pipeline_midi import MIDIPipeline

        print(f"[MIDI] 加载 pipeline 从 {MIDI_LOCAL_DIR} ...")
        pipe = MIDIPipeline.from_pretrained(
            MIDI_LOCAL_DIR,
            torch_dtype=torch.float16,
        ).to(DEVICE)

        # 确保所有子模块 dtype 一致 (修复 CLIP bf16/fp16 混用)
        pipe = pipe.to(dtype=torch.float16)

        # 初始化 multi-instance attention adapter (最后 5 层 self-attn)
        pipe.init_custom_adapter(
            set_self_attn_module_names=[
                "blocks.8", "blocks.9", "blocks.10", "blocks.11", "blocks.12",
            ]
        )
        print(f"[MIDI] Pipeline 加载完成 (device={DEVICE}).")

        _midi_pipeline = pipe

    except ImportError as e:
        warnings.warn(
            f"[MIDI] 依赖缺失: {e}\n"
            "  Install: pip install diffusers trimesh scikit-image peft"
        )
    except Exception as e:
        warnings.warn(f"[MIDI] Pipeline 加载失败: {e}")
        import traceback
        traceback.print_exc()

    return _midi_pipeline


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def masks_to_segmentation(masks: List[np.ndarray]) -> np.ndarray:
    """将逐物体二值 mask 合并为多标签分割图 (label 0=背景, 1..N=物体)."""
    h, w = masks[0].shape[:2]
    seg = np.zeros((h, w), dtype=np.uint8)
    for i, m in enumerate(masks):
        seg[m > 0] = i + 1
    return seg


def extract_instance_images(
    image: np.ndarray, masks: List[np.ndarray]
) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image], Image.Image]:
    """从场景图像 + mask 列表提取 MIDI 所需的三组输入.

    Returns:
        instance_rgbs:   白色背景 + 物体像素 (每物体一张)
        instance_masks:  二值 mask (255=物体)
        scene_rgbs:      完整场景图像 (每物体复制一份)
        seg_pil:         多标签分割图 PIL
    """
    rgb_pil = Image.fromarray(image)
    seg_np = masks_to_segmentation(masks)
    seg_pil = Image.fromarray(seg_np.astype(np.uint8))

    instance_rgbs, instance_masks, scene_rgbs = [], [], []
    for mask in masks:
        white_bg = np.ones_like(image) * 255
        obj_rgb = white_bg.copy()
        obj_rgb[mask > 0] = image[mask > 0]
        instance_rgbs.append(Image.fromarray(obj_rgb))
        instance_masks.append(Image.fromarray((mask * 255).astype(np.uint8)))
        scene_rgbs.append(rgb_pil)

    return instance_rgbs, instance_masks, scene_rgbs, seg_pil


# ---------------------------------------------------------------------------
# MIDIWrapper 主类
# ---------------------------------------------------------------------------

class MIDIWrapper:
    """MIDI 多实例 3D 生成封装器 — 兼容 SAM3DWrapper 接口."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipe = None
        self._available = False
        self._init_model()

    def _init_model(self):
        try:
            self._pipe = _get_midi_pipeline()
            self._available = self._pipe is not None
        except Exception as e:
            warnings.warn(f"[MIDIWrapper] 模型加载失败: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @torch.no_grad()
    def generate_from_scene(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        bboxes: List[Tuple[int, int, int, int]] = None,
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
    ) -> Tuple[List, List[Dict]]:
        """从场景图像 + mask 列表同时生成所有物体的 3D mesh.

        Args:
            image:   (H, W, 3) uint8 RGB 场景图像
            masks:   list of (H, W) uint8 binary masks (0/1)
            seed:    随机种子
            num_inference_steps: 去噪步数
            guidance_scale: CFG scale

        Returns:
            meshes:     Open3D TriangleMesh 列表 (已在场景坐标系中)
            transforms: dict 列表 {'R': 3x3, 't': 3, 's': 1.0} (恒等变换)
        """
        import open3d as o3d
        import trimesh
        from skimage import measure
        from midi.utils.smoothing import smooth_gpu

        pipe = self._pipe
        if pipe is None:
            raise RuntimeError("[MIDIWrapper] Pipeline 未加载")

        # 1. 准备 MIDI 输入
        instance_rgbs, instance_masks, scene_rgbs, _ = extract_instance_images(
            image, masks
        )
        N = len(instance_rgbs)
        print(f"[MIDI] 批量生成 {N} 个物体 (steps={num_inference_steps}) ...")

        # 2. MIDI 推理
        generator = torch.Generator(device=self.device).manual_seed(seed)
        outputs = pipe(
            image=instance_rgbs,
            mask=instance_masks,
            image_scene=scene_rgbs,
            attention_kwargs={"num_instances": N},
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            decode_progressive=True,
            return_dict=False,
        )

        # 3. Marching Cubes -> 每物体 mesh
        meshes = []
        transforms = []
        for i, (logits_, grid_size, bbox_size, bbox_min, bbox_max) in enumerate(
            zip(*outputs)
        ):
            grid_logits = logits_.view(grid_size)
            grid_logits = smooth_gpu(grid_logits, method="gaussian", sigma=1)
            torch.cuda.empty_cache()

            logits_np = grid_logits.float().cpu().numpy()
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    logits_np, 0, method="lewiner"
                )
            except (RuntimeError, ValueError):
                print(f"  [{i}] warning: marching_cubes 未找到表面，生成空 mesh")
                meshes.append(o3d.geometry.TriangleMesh())
                transforms.append(
                    {"R": np.eye(3).tolist(), "t": [0, 0, 0], "s": 1.0}
                )
                continue

            verts = verts / grid_size * bbox_size + bbox_min
            tri_mesh = trimesh.Trimesh(
                verts.astype(np.float32), np.ascontiguousarray(faces)
            )
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(
                np.array(tri_mesh.vertices)
            )
            o3d_mesh.triangles = o3d.utility.Vector3iVector(
                np.array(tri_mesh.faces)
            )
            o3d_mesh.compute_vertex_normals()

            meshes.append(o3d_mesh)
            transforms.append(
                {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "s": 1.0}
            )
            print(f"  [{i}] {len(tri_mesh.vertices)} verts, {len(tri_mesh.faces)} faces")

        return meshes, transforms

    # ---- 向后兼容 SAM3DWrapper.generate() ----

    @torch.no_grad()
    def generate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        output_format: str = "mesh",
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
    ) -> Tuple:
        """单物体生成 — 兼容 SAM3DWrapper.generate() 接口."""
        meshes, transforms = self.generate_from_scene(
            image=image,
            masks=[mask],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        mesh = meshes[0]
        T = transforms[0]
        return mesh, np.array(T["R"]), np.array(T["t"]), float(T["s"])
