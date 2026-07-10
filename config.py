"""
CAST configuration and default parameters.

Many settings mirror the paper's reported values. Tune for your hardware
and quality/speed trade-off.
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class CASTConfig:
    """Top-level configuration for the CAST pipeline."""

    # ---- Scene Analysis ----
    # Object detection / segmentation
    detection_model: str = "florence-2"  # florence-2 | grounding-dino
    depth_model: str = "moge"            # moge | metric3d | zoedepth
    sam_variant: str = "sam2"            # sam2 | sam (GroundedSAMv2)

    # ---- VLM for relation reasoning (Section 5.3) ----
    # Provider selection: "qwen" | "openai" | None (skip)
    # Both use OpenAI-compatible API format.
    vlm_provider: Optional[str] = "qwen"

    # Qwen / DashScope settings
    # Model choices: qwen-vl-max, qwen-vl-plus, qwen2.5-vl-72b-instruct
    qwen_api_key: Optional[str] = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-vl-max"

    # OpenAI / GPT-4V settings (fallback if vlm_provider == "openai")
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    gpt_model: str = "gpt-4-vision-preview"

    # VLM ensemble settings (shared)
    vlm_ensemble_trials: int = 3         # majority-vote trials
    vlm_max_tokens: int = 2048
    vlm_temperature: float = 0.3

    # ---- SAM 3D Object Generation ----
    # Replaces the original ObjectGen (CLAY-based) + AlignGen.
    # Uses Meta SAM 3D Objects (facebook/sam-3d-objects) for
    # one-shot mesh + texture + 6D pose prediction.
    sam3d_model_id: str = "facebook/sam-3d-objects"
    sam3d_use_fp16: bool = True           # half-precision saves ~3.5 GB VRAM
    sam3d_offline: bool = False           # skip HuggingFace download attempt
    sam3d_inference_steps: int = 50       # flow matching steps
    sam3d_guidance_scale: float = 3.0     # CFG scale

    # ---- Pose Adapter ----
    # Bridges SAM 3D camera-space pose → CAST scene-space coordinates.
    pose_refinement: str = "icp"          # "none" | "umeyama" | "icp"
    icp_max_distance: float = 0.1         # max correspondence distance (meters)

    # ---- Point-cloud (shared) ----
    pc_num_points: int = 2048             # FPS-sampled points for conditioning/refinement

    # ---- Iterative Refinement (kept for config compat; SAM 3D is one-shot) ----
    max_iterations: int = 1
    convergence_threshold: float = 0.01

    # ---- Physics-Aware Correction ----
    enable_physics_correction: bool = True
    physics_surface_samples: int = 2048   # surface points sampled per object
    physics_sdf_sigma: float = 0.05       # near-surface threshold (paper: sigma)
    physics_optim_steps: int = 200
    physics_lr: float = 0.01

    # ---- I/O ----
    output_dir: str = "./output"
    device: str = "cuda"

    # ------------------------------------------------------------------
    def get_vlm_api_key(self) -> Optional[str]:
        """Return the active VLM API key based on provider choice."""
        if self.vlm_provider == "qwen":
            return self.qwen_api_key
        elif self.vlm_provider == "openai":
            return self.openai_api_key
        return None

    def get_vlm_base_url(self) -> Optional[str]:
        """Return the active VLM base URL."""
        if self.vlm_provider == "qwen":
            return self.qwen_base_url
        elif self.vlm_provider == "openai":
            return self.openai_base_url or "https://api.openai.com/v1"
        return None

    def get_vlm_model(self) -> str:
        """Return the active VLM model name."""
        if self.vlm_provider == "qwen":
            return self.qwen_model
        elif self.vlm_provider == "openai":
            return self.gpt_model
        return ""

    @property
    def vlm_enabled(self) -> bool:
        """Whether VLM relation reasoning is available."""
        return (self.vlm_provider is not None
                and self.get_vlm_api_key() is not None)


# Convenient presets
def quick_config() -> CASTConfig:
    """Return a config tuned for fast prototyping (lower quality)."""
    return CASTConfig(
        sam3d_inference_steps=20,
        pose_refinement="umeyama",
        physics_optim_steps=50,
        enable_physics_correction=False,
    )


def full_config() -> CASTConfig:
    """Return the full-quality config."""
    return CASTConfig()


def qwen_config(api_key: str) -> CASTConfig:
    """Return a config pre-configured for Qwen (DashScope) VLM."""
    return CASTConfig(
        vlm_provider="qwen",
        qwen_api_key=api_key,
    )
