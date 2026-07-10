"""
Scene analysis: object detection, segmentation, depth estimation, and camera
parameter extraction from a single RGB image.

Implements the preprocessing stage described in Section 3 (Overview) of CAST:

1. Florence-2 → object descriptions + bounding boxes
2. VLM (Qwen / GPT-4V) → filter spurious detections, build open-vocabulary object list
3. GroundedSAMv2 → refined per-object segmentation masks {M_i}
4. MoGe → pixel-aligned point clouds {q_i} + global camera parameters

NOTE:
    The full CAST pipeline uses several large foundation models that require
    separate setup. This module provides:
    - A unified interface for swapping backends (Florence-2 / Grounding DINO /
      SAM / MoGe / Depth-Anything-V2)
    - Lazy loading for all models (only downloaded on first use)
    - Graceful fallback to stub/placeholder when models are unavailable
    - Full VLM integration (Qwen DashScope / GPT-4V OpenAI) via utils/relation_graph.py
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class ObjectInfo:
    """Per-object metadata extracted during scene analysis."""
    id: int
    name: str                       # e.g. "chair", "table"
    description: str                 # e.g. "wooden chair with armrests"
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords
    mask: np.ndarray                 # H x W binary mask (uint8, 0/1)
    point_cloud: np.ndarray          # N x 3 in camera space (from depth)
    occlusion_mask: np.ndarray       # H x W binary mask of visible pixels


@dataclass
class SceneAnalysisResult:
    """Output of the scene-analysis stage."""
    image: np.ndarray                # original RGB image (H x W x 3)
    objects: List[ObjectInfo]        # per-object data
    depth_map: np.ndarray            # H x W metric depth
    camera_intrinsics: np.ndarray    # 3 x 3 camera intrinsics
    camera_pose: np.ndarray          # 4 x 4 world-to-camera (identity if unknown)


# ============================================================================
# Shared utilities
# ============================================================================

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 使用 fp16 的 dtype（GPU 可以节省显存）
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ============================================================================
# Lazy loader: Florence-2 (open-vocabulary detection)
# ============================================================================

FLORENCE2_MODEL_NAME = "microsoft/Florence-2-large"

_florence2_model = None
_florence2_processor = None


def _get_florence2_model():
    """懒加载 Florence-2 模型与处理器（全局单例）."""
    global _florence2_model, _florence2_processor
    if _florence2_model is None:
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            _florence2_processor = AutoProcessor.from_pretrained(
                FLORENCE2_MODEL_NAME, trust_remote_code=True
            )
            _florence2_model = AutoModelForCausalLM.from_pretrained(
                FLORENCE2_MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=DTYPE,
            ).to(DEVICE)
            _florence2_model.eval()
            print(f"[SceneAnalysis] Florence-2 loaded on {DEVICE}.")
        except Exception as e:
            warnings.warn(
                f"[SceneAnalysis] Florence-2 模型加载失败: {e}\n"
                f"  Install: pip install transformers\n"
                f"  Model: {FLORENCE2_MODEL_NAME} (~1.7 GB)"
            )
    return _florence2_model, _florence2_processor


# ============================================================================
# Lazy loader: Grounding DINO (open-set detection, fallback for Florence-2)
# ============================================================================

GDINO_MODEL_NAME = "IDEA-Research/grounding-dino-tiny"

_gdino_model = None
_gdino_processor = None


def _get_grounding_dino_model():
    """懒加载 Grounding DINO 模型与处理器（全局单例）."""
    global _gdino_model, _gdino_processor
    if _gdino_model is None:
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            _gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_NAME)
            _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GDINO_MODEL_NAME
            ).to(DEVICE)
            _gdino_model.eval()
            print(f"[SceneAnalysis] Grounding DINO loaded on {DEVICE}.")
        except Exception as e:
            warnings.warn(
                f"[SceneAnalysis] Grounding DINO 模型加载失败: {e}\n"
                f"  Install: pip install transformers\n"
                f"  Model: {GDINO_MODEL_NAME} (~700 MB)"
            )
    return _gdino_model, _gdino_processor


# ============================================================================
# Lazy loader: SAM (bbox-prompted mask generation)
# ============================================================================

SAM_MODEL_NAME = "facebook/sam-vit-base"

_sam_model = None
_sam_processor = None


def _get_sam_model():
    """懒加载 SAM 模型与处理器（全局单例）."""
    global _sam_model, _sam_processor
    if _sam_model is None:
        try:
            from transformers import AutoModelForMaskGeneration, AutoProcessor
            _sam_processor = AutoProcessor.from_pretrained(SAM_MODEL_NAME)
            _sam_model = AutoModelForMaskGeneration.from_pretrained(
                SAM_MODEL_NAME
            ).to(DEVICE)
            _sam_model.eval()
            print(f"[SceneAnalysis] SAM loaded on {DEVICE}.")
        except Exception as e:
            warnings.warn(
                f"[SceneAnalysis] SAM 模型加载失败: {e}\n"
                f"  Install: pip install transformers torchvision\n"
                f"  Model: {SAM_MODEL_NAME} (~1.2 GB)"
            )
    return _sam_model, _sam_processor


# ============================================================================
# Lazy loader: MoGe (metric monocular depth)
# ============================================================================

MOGE_MODEL_NAME = "Ruicheng/moge-vitl"

_moge_model = None


def _get_moge_model():
    """懒加载 MoGe 模型（全局单例）."""
    global _moge_model
    if _moge_model is None:
        try:
            # Try v2 first, fall back to v1 for older models
            try:
                from moge.model.v2 import MoGeModel as _MoGeModel
            except ImportError:
                from moge.model.v1 import MoGeModel as _MoGeModel
            _moge_model = _MoGeModel.from_pretrained(MOGE_MODEL_NAME).to(DEVICE)
            _moge_model.eval()
            print(f"[SceneAnalysis] MoGe loaded on {DEVICE}.")
        except ImportError:
            warnings.warn(
                "[SceneAnalysis] MoGe 未安装。\n"
                "  Install: pip install moge\n"
                "  (or clone https://github.com/microsoft/MoGe)"
            )
        except (TypeError, AttributeError) as e:
            # v2 code + v1 model mismatch — try v1 explicitly
            try:
                from moge.model.v1 import MoGeModel as _MoGeModel_v1
                _moge_model = _MoGeModel_v1.from_pretrained(MOGE_MODEL_NAME).to(DEVICE)
                _moge_model.eval()
                print(f"[SceneAnalysis] MoGe (v1) loaded on {DEVICE}.")
            except Exception as e2:
                warnings.warn(
                    f"[SceneAnalysis] MoGe 模型加载失败: {e2}\n"
                    f"  Model: {MOGE_MODEL_NAME} (~1.2 GB)"
                )
        except Exception as e:
            warnings.warn(
                f"[SceneAnalysis] MoGe 模型加载失败: {e}\n"
                f"  Model: {MOGE_MODEL_NAME} (~1.2 GB)"
            )
    return _moge_model


# ============================================================================
# Lazy loader: Depth-Anything-V2 (alternative monocular depth)
# ============================================================================

DAV2_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

_dav2_model = None
_dav2_processor = None


def _get_depth_anything_v2_model():
    """懒加载 Depth-Anything-V2 模型与处理器（全局单例）."""
    global _dav2_model, _dav2_processor
    if _dav2_model is None:
        try:
            from transformers import AutoModelForDepthEstimation, AutoImageProcessor
            _dav2_processor = AutoImageProcessor.from_pretrained(DAV2_MODEL_NAME)
            _dav2_model = AutoModelForDepthEstimation.from_pretrained(
                DAV2_MODEL_NAME
            ).to(DEVICE)
            _dav2_model.eval()
            print(f"[SceneAnalysis] Depth-Anything-V2 loaded on {DEVICE}.")
        except Exception as e:
            warnings.warn(
                f"[SceneAnalysis] Depth-Anything-V2 模型加载失败: {e}\n"
                f"  Install: pip install transformers\n"
                f"  Model: {DAV2_MODEL_NAME} (~200 MB)"
            )
    return _dav2_model, _dav2_processor


# ============================================================================
# Backend: object detection
# ============================================================================

def detect_objects_florence2(image: np.ndarray) -> List[Dict]:
    """
    Use Florence-2 for open-vocabulary object detection.

    REQUIRES EXTERNAL SETUP:
        pip install transformers torch pillow numpy accelerate
        Model: microsoft/Florence-2-large  (~1.7 GB download)

    Args:
        image: HWC 格式 uint8 numpy 数组 (RGB)
    Returns list of dicts with keys: 'name', 'description', 'bbox'.
        bbox: [x1, y1, x2, y2] 像素坐标 (int)
    """
    model, processor = _get_florence2_model()

    if model is None or processor is None:
        warnings.warn(
            "detect_objects_florence2: model not loaded, returning empty list."
        )
        return []

    # 1. numpy array -> PIL Image
    pil_img = Image.fromarray(image)

    # 2. Open-vocabulary detection task prompt
    prompt = "<OD>"

    # 3. Preprocess
    inputs = processor(
        text=prompt, images=pil_img, return_tensors="pt"
    )
    # Only convert pixel_values to model dtype; input_ids must stay Long
    inputs["pixel_values"] = inputs["pixel_values"].to(DEVICE, DTYPE)
    inputs["input_ids"] = inputs["input_ids"].to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
        )

    # 5. Decode & parse detection boxes
    # NOTE: skip_special_tokens=False is REQUIRED — <loc_N> tokens encode
    # bounding-box coordinates and must be preserved for post-processing.
    result_text = processor.decode(outputs[0], skip_special_tokens=False)
    parsed_objects = processor.post_process_generation(
        result_text,
        task="<OD>",
        image_size=(pil_img.width, pil_img.height),
    )
    od_results = parsed_objects["<OD>"]

    # 6. Format output
    output_list = []
    for label, bbox in zip(od_results["labels"], od_results["bboxes"]):
        x1, y1, x2, y2 = bbox
        output_list.append({
            "name": label,
            "description": f"detected {label}",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    return output_list


def detect_objects_grounding_dino(image: np.ndarray,
                                  text_prompt: str = "object") -> List[dict]:
    """
    Use Grounding DINO for open-set object detection.

    REQUIRES EXTERNAL SETUP:
        pip install transformers torch
        Model: IDEA-Research/grounding-dino-tiny  (~700 MB download)
              or IDEA-Research/grounding-dino-base  (~700 MB download)

    Args:
        image:       HWC 格式 uint8 numpy 数组 (RGB)
        text_prompt: 文本提示词，描述要检测的目标，如 "chair . table . lamp"
                     用 " . " 分隔多个类别

    Returns list of dicts with keys: 'name', 'description', 'bbox'.
        bbox: [x1, y1, x2, y2] 像素坐标 (int)
    """
    model, processor = _get_grounding_dino_model()

    if model is None or processor is None:
        warnings.warn(
            "detect_objects_grounding_dino: model not loaded, returning empty list."
        )
        return []

    # 1. numpy array -> PIL Image
    pil_img = Image.fromarray(image)

    # 2. Normalise text prompt: Grounding DINO uses " . " as class separator
    normalized_prompt = text_prompt.strip().lower()
    if " . " not in normalized_prompt and "." in normalized_prompt:
        normalized_prompt = normalized_prompt.replace(".", " . ")
    # Ensure trailing "." (Grounding DINO convention)
    if not normalized_prompt.endswith("."):
        normalized_prompt = normalized_prompt + " ."

    # 3. Preprocess
    inputs = processor(
        images=pil_img, text=normalized_prompt, return_tensors="pt"
    ).to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Post-process: parse logits + boxes → detection results
    target_sizes = torch.tensor([[pil_img.height, pil_img.width]]).to(DEVICE)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.25,        # detection confidence threshold
        text_threshold=0.25,   # text matching threshold
        target_sizes=target_sizes,
    )

    # 6. Format output
    if not results:
        return []

    result = results[0]  # single-image result
    output_list = []
    for score, label, box in zip(
        result.get("scores", []),
        result.get("labels", []),
        result.get("boxes", []),
    ):
        x1, y1, x2, y2 = box.tolist()
        output_list.append({
            "name": label,
            "description": f"detected {label} ({score:.2f})",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    return output_list


# ============================================================================
# Backend: segmentation
# ============================================================================

def segment_objects_grounded_sam(image: np.ndarray,
                                 bboxes: List[Tuple[int, int, int, int]],
                                 labels: List[str]) -> List[np.ndarray]:
    """
    GroundedSAMv2: generate per-object segmentation masks from bounding boxes.

    Uses SAM (Segment Anything Model) with bbox-prompting. Each bounding box
    is used as a spatial prompt, and the model predicts a refined mask.

    REQUIRES EXTERNAL SETUP:
        pip install transformers torchvision
        Model: facebook/sam-vit-base  (~1.2 GB download)
              or facebook/sam-vit-huge (~2.4 GB, better quality)

    Args:
        image:  (H, W, 3) uint8 RGB image
        bboxes: list of (x1, y1, x2, y2) pixel coordinates
        labels: list of object names (for logging only)

    Returns:
        List of (H, W) uint8 binary masks, one per bounding box.
    """
    model, processor = _get_sam_model()

    if model is None or processor is None:
        warnings.warn(
            "segment_objects_grounded_sam: model not loaded, "
            "falling back to bbox-only masks."
        )
        # Fallback: use bbox as rectangular mask
        h, w = image.shape[:2]
        masks = []
        for (x1, y1, x2, y2) in bboxes:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)
        return masks

    # 1. numpy array -> PIL Image
    pil_img = Image.fromarray(image)

    # 2. Prepare bboxes in the format expected by SamProcessor:
    #    list of lists: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
    input_boxes = [[float(x1), float(y1), float(x2), float(y2)]
                   for (x1, y1, x2, y2) in bboxes]

    # 3. Preprocess image AND boxes together — the processor rescales
    #    bbox coordinates to match the resized image fed to the model
    inputs = processor(
        images=pil_img,
        input_boxes=[input_boxes],  # nested: [batch_list]
        return_tensors="pt",
    ).to(DEVICE)

    # 4. Inference — SAM generates masks from bbox prompts
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # 5. Post-process masks using the processor's built-in method
    #    (handles resize + thresholding back to original image size)
    pred_masks = outputs.pred_masks  # (1, N, 1, H_mask, W_mask)
    masks_np = processor.post_process_masks(
        pred_masks,
        original_sizes=inputs["original_sizes"],
        reshaped_input_sizes=inputs["reshaped_input_sizes"],
        binarize=True,           # threshold at 0.0
    )[0]  # first (and only) batch item

    # Convert to list of uint8 arrays: each mask is (1, H, W)
    # post_process_masks returns (1, H, W) when multimask_output=False
    masks = []
    for mask_tensor in masks_np:
        # Take first channel and convert to uint8
        mask = mask_tensor[0].cpu().numpy().astype(np.uint8)
        masks.append(mask)

    return masks


# ============================================================================
# Backend: monocular depth estimation
# ============================================================================

def estimate_depth_moge(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MoGe (Wang et al. 2024) for metric monocular depth estimation.

    MoGe predicts both metric depth and camera intrinsics from a single RGB
    image. It is the primary depth backend used in the CAST paper.

    REQUIRES EXTERNAL SETUP:
        pip install moge
        (or clone https://github.com/microsoft/MoGe)
        Model: Ruicheng/moge-vitl  (~1.2 GB download)

    Args:
        image: (H, W, 3) uint8 RGB image

    Returns:
        depth_map:  (H, W) metric depth in meters
        intrinsics: (3, 3) camera intrinsics estimated by MoGe
    """
    model = _get_moge_model()

    if model is None:
        warnings.warn(
            "estimate_depth_moge: model not loaded, using placeholder "
            "(constant 2m depth)."
        )
        h, w = image.shape[:2]
        depth_map = np.full((h, w), 2.0, dtype=np.float32)
        fx = fy = max(w, h)
        cx, cy = w / 2.0, h / 2.0
        intrinsics = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )
        return depth_map, intrinsics

    # 1. numpy array -> PIL Image (for v2) / tensor (for v1)
    pil_img = Image.fromarray(image)

    # 2. Run MoGe inference
    with torch.no_grad():
        try:
            # v2: accepts PIL Image
            output = model.infer(pil_img)
        except (AttributeError, TypeError):
            # v1: accepts tensor (C, H, W) in [0, 1]
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(DEVICE)
            output = model.infer(img_tensor)

    # 3. Extract depth and intrinsics
    depth_map = output["depth"].squeeze().cpu().numpy().astype(np.float32)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy().astype(np.float32)

    # MoGe returns normalized intrinsics — convert to pixel coordinates
    h, w = image.shape[:2]
    intrinsics[0, 0] *= w   # fx
    intrinsics[1, 1] *= h   # fy
    intrinsics[0, 2] *= w   # cx
    intrinsics[1, 2] *= h   # cy

    return depth_map, intrinsics


def estimate_depth_metric3d(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alternative: Depth-Anything-V2 for relative depth estimation.

    Uses the HuggingFace Depth-Anything-V2 pipeline. Note that Depth-Anything
    produces relative (not metric) depth — we apply a heuristic to convert
    to approximate meters (assuming scene depth range ~0.5-10m).

    REQUIRES EXTERNAL SETUP:
        pip install transformers
        Model: depth-anything/Depth-Anything-V2-Small-hf  (~200 MB)
              or depth-anything/Depth-Anything-V2-Large-hf (~1.3 GB)

    Args:
        image: (H, W, 3) uint8 RGB image

    Returns:
        depth_map:  (H, W) approximate metric depth in meters
        intrinsics: (3, 3) pinhole camera intrinsics (estimated heuristically)
    """
    model, processor = _get_depth_anything_v2_model()

    if model is None or processor is None:
        warnings.warn(
            "estimate_depth_metric3d: model not loaded, using placeholder "
            "(constant 2m depth)."
        )
        h, w = image.shape[:2]
        depth_map = np.full((h, w), 2.0, dtype=np.float32)
        fx = fy = max(w, h)
        intrinsics = np.array(
            [[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float32
        )
        return depth_map, intrinsics

    # 1. numpy array -> PIL Image
    pil_img = Image.fromarray(image)

    # 2. Preprocess
    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)

    # 3. Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Post-process
    pred = outputs.predicted_depth  # (1, H, W) — relative depth

    # Resize to original image size
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(image.shape[0], image.shape[1]), mode="bicubic"
    ).squeeze()

    depth_relative = pred.cpu().numpy().astype(np.float32)

    # Normalise to [0, 1] then map to approximate metric range
    d_min, d_max = depth_relative.min(), depth_relative.max()
    if d_max - d_min > 1e-6:
        depth_normalised = (depth_relative - d_min) / (d_max - d_min)
    else:
        depth_normalised = depth_relative

    # Heuristic: map [0,1] → [0.5, 10.0] meters
    depth_map = 0.5 + depth_normalised * 9.5

    # Pinhole intrinsics (heuristic: FoV ~60 degrees)
    h, w = image.shape[:2]
    fx = fy = max(w, h) / (2.0 * np.tan(np.deg2rad(30.0)))
    cx, cy = w / 2.0, h / 2.0
    intrinsics = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
    )

    return depth_map, intrinsics


# ============================================================================
# Depth → point cloud
# ============================================================================

def depth_to_point_cloud(depth_map: np.ndarray,
                         intrinsics: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Unproject depth map to 3D point cloud in camera space.

    Args:
        depth_map:  (H, W) metric depth in meters
        intrinsics: (3, 3) camera intrinsics matrix
        mask:       (H, W) binary mask (optional); only unproject masked pixels

    Returns:
        (N, 3) point cloud in camera space
    """
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    v, u = np.mgrid[0:h, 0:w]
    z = depth_map.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    if mask is not None:
        pts = pts[mask > 0]
        # Filter invalid depths within the masked region
        pts = pts[z[mask > 0] > 0.01]
    else:
        pts = pts.reshape(-1, 3)
        pts = pts[z.reshape(-1) > 0.01]

    return pts


# ============================================================================
# VLM relation reasoning (Section 5.3) — Qwen / GPT-4V via utils/relation_graph
# ============================================================================

def build_relation_graph_vlm(image: np.ndarray,
                              objects: List[ObjectInfo],
                              vlm_provider: Optional[str] = None,
                              api_key: Optional[str] = None,
                              base_url: Optional[str] = None,
                              model: Optional[str] = None,
                              ensemble_trials: int = 3,
                              max_tokens: int = 2048,
                              temperature: float = 0.3) -> Dict:
    """
    Use VLM (Qwen or GPT-4V) + Set-of-Mark to extract pairwise physical relations.

    Implements the method described in Section 5.3:
      1) Annotate image with SoM (numbered masks).
      2) Query VLM with the prompt from Listing 1.
      3) Ensemble across trials (majority voting, >= half).
      4) Map fine-grained relations -> Support / Contact.

    Args:
        image:           (H, W, 3) uint8 RGB image
        objects:         list of ObjectInfo
        vlm_provider:    "qwen" or "openai" (None = skip)
        api_key:         VLM API key
        base_url:        VLM API base URL
        model:           VLM model name
        ensemble_trials: number of independent trials for majority voting
        max_tokens:      max output tokens per call
        temperature:     sampling temperature

    Returns a relation graph dict:
        {
            "nodes": [0, 1, ..., N-1],
            "contact_edges": [(i, j), ...],
            "support_edges": [(supporter, supported), ...]
        }
    """
    if api_key is None:
        warnings.warn(
            "[SceneAnalysis] No VLM API key provided. "
            "Returning empty relation graph."
        )
        return {"nodes": list(range(len(objects))),
                "contact_edges": [],
                "support_edges": []}

    if vlm_provider is None:
        warnings.warn(
            "[SceneAnalysis] vlm_provider not set. "
            "Returning empty relation graph."
        )
        return {"nodes": list(range(len(objects))),
                "contact_edges": [],
                "support_edges": []}

    from utils.relation_graph import query_vlm_with_som

    print(f"\n[SceneAnalysis] Querying VLM ({vlm_provider}: {model}) "
          f"for relation graph ...")

    return query_vlm_with_som(
        image=image,
        objects=objects,
        api_key=api_key,
        base_url=base_url,
        model=model,
        ensemble_trials=ensemble_trials,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=True,
    )


# Backward compatibility alias
build_relation_graph_gpt4v = build_relation_graph_vlm


# ============================================================================
# Top-level scene-analysis pipeline
# ============================================================================

def analyze_scene(image: np.ndarray,
                  openai_api_key: Optional[str] = None,
                  openai_base_url: Optional[str] = None,
                  gpt_model: str = "gpt-4-vision-preview",
                  # New VLM params (preferred over the OpenAI-specific ones above)
                  vlm_provider: Optional[str] = None,
                  vlm_api_key: Optional[str] = None,
                  vlm_base_url: Optional[str] = None,
                  vlm_model: Optional[str] = None,
                  vlm_ensemble_trials: int = 3,
                  vlm_max_tokens: int = 2048,
                  vlm_temperature: float = 0.3,
                  ) -> SceneAnalysisResult:
    """
    Run the full scene-analysis preprocessing pipeline (Section 3).

    Stages:
      1. Object detection (Florence-2)
      2. VLM filtering (optional)
      3. Segmentation (GroundedSAMv2)
      4. Depth estimation (MoGe)
      5. Per-object point cloud extraction

    Args:
        image:             RGB image (H x W x 3, uint8)
        openai_api_key:    (deprecated) GPT-4V API key
        openai_base_url:   (deprecated) API base URL override
        gpt_model:         (deprecated) GPT model name
        vlm_provider:      "qwen" | "openai" | None
        vlm_api_key:       VLM API key
        vlm_base_url:      VLM API base URL
        vlm_model:         VLM model name
        vlm_ensemble_trials: number of majority-vote trials
        vlm_max_tokens:    max output tokens per VLM call
        vlm_temperature:   VLM sampling temperature

    Returns:
        SceneAnalysisResult with objects, depth, camera info.
    """
    h, w = image.shape[:2]

    # Resolve VLM settings: new params take precedence over deprecated ones
    _vlm_provider = vlm_provider
    _vlm_api_key = vlm_api_key
    _vlm_base_url = vlm_base_url
    _vlm_model = vlm_model

    # Fallback: if old-style openai params are set and new params are not
    if _vlm_provider is None and openai_api_key is not None:
        _vlm_provider = "openai"
        _vlm_api_key = openai_api_key
        _vlm_base_url = openai_base_url
        _vlm_model = gpt_model

    # 1. Detect objects with Florence-2
    detections = detect_objects_florence2(image)

    # 2. VLM filtering (optional — can refine Florence-2 detections with VLM)
    #    Currently skipped; VLM is used only for relation graph in Stage 3.
    #    To enable: uncomment below and pass _vlm_api_key, _vlm_model, etc.

    # 3. Refine masks with GroundedSAMv2
    if detections:
        bboxes = [d["bbox"] for d in detections]
        labels = [d.get("name", "object") for d in detections]
        masks = segment_objects_grounded_sam(image, bboxes, labels)
    else:
        # Fallback: treat entire image as one object
        detections = [{
            "name": "scene", "description": "full scene",
            "bbox": (0, 0, w, h),
        }]
        masks = [np.ones((h, w), dtype=np.uint8)]

    # 4. Metric depth estimation (MoGe)
    depth_map, intrinsics = estimate_depth_moge(image)
    camera_pose = np.eye(4, dtype=np.float32)  # camera at origin

    # Occlusion mask: for each object, subtract other objects' masks
    occlusion_masks = []
    for i, m in enumerate(masks):
        occ = m.copy()
        for j, other_m in enumerate(masks):
            if j != i:
                occ[other_m > 0] = 0
        occlusion_masks.append(occ)

    # 5. Per-object point clouds
    objects = []
    for i, det in enumerate(detections):
        pc = depth_to_point_cloud(depth_map, intrinsics, masks[i])
        obj = ObjectInfo(
            id=i,
            name=det.get("name", f"object_{i}"),
            description=det.get("description", ""),
            bbox=det.get("bbox", (0, 0, w, h)),
            mask=masks[i],
            point_cloud=pc,
            occlusion_mask=occlusion_masks[i],
        )
        objects.append(obj)

    return SceneAnalysisResult(
        image=image,
        objects=objects,
        depth_map=depth_map,
        camera_intrinsics=intrinsics,
        camera_pose=camera_pose,
    )
