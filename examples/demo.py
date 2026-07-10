"""
Minimal demo of the CAST pipeline.

Usage:
    # With Qwen VLM (recommended default):
    python examples/demo.py --image path/to/image.jpg --qwen-key sk-xxxxx

    # With GPT-4V:
    python examples/demo.py --image path/to/image.jpg --vlm openai --openai-key sk-xxxxx

    # Quick test without any VLM (skips relation graph + physics):
    python examples/demo.py --image path/to/image.jpg --quick --device cpu

Notes:
  - SAM 3D Objects (facebook/sam-3d-objects) is downloaded automatically
    from HuggingFace on first run (~7 GB). Set --sam3d-offline to skip.
  - Without a VLM API key, the relation graph is empty and physics
    correction is skipped.
  - See README.md for setup instructions.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
from config import CASTConfig, quick_config, qwen_config
from pipeline import CASTPipeline


def main():
    parser = argparse.ArgumentParser(
        description="CAST: Single-Image 3D Scene Reconstruction (SAM 3D + Qwen VLM)")

    # Image
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input RGB image')
    parser.add_argument('--output', type=str, default='./output/demo',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Use quick (lower-quality) config')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda / cpu)')

    # VLM provider selection
    parser.add_argument('--vlm', type=str, default='qwen',
                        choices=['qwen', 'openai'],
                        help='VLM provider for relation graph (default: qwen)')
    parser.add_argument('--qwen-key', type=str, default=None,
                        help='Qwen (DashScope) API key')
    parser.add_argument('--qwen-base-url', type=str,
                        default='https://dashscope.aliyuncs.com/compatible-mode/v1',
                        help='Qwen API base URL')
    parser.add_argument('--qwen-model', type=str, default='qwen-vl-max',
                        help='Qwen model name')
    parser.add_argument('--openai-key', type=str, default=None,
                        help='OpenAI API key (for GPT-4V fallback)')
    parser.add_argument('--openai-base-url', type=str, default=None,
                        help='OpenAI API base URL')
    parser.add_argument('--openai-model', type=str, default='gpt-4-vision-preview',
                        help='GPT-4V model name')
    parser.add_argument('--vlm-trials', type=int, default=3,
                        help='VLM ensemble trials for majority voting')

    # SAM 3D options
    parser.add_argument('--sam3d-model-id', type=str,
                        default='facebook/sam-3d-objects',
                        help='HuggingFace model ID for SAM 3D')
    parser.add_argument('--sam3d-offline', action='store_true',
                        help='Skip SAM 3D download; use stub fallback')
    parser.add_argument('--sam3d-fp32', action='store_true',
                        help='Use full precision (instead of fp16)')

    # Pose adapter options
    parser.add_argument('--pose-refinement', type=str, default='icp',
                        choices=['none', 'umeyama', 'icp'],
                        help='Pose refinement method (default: icp)')

    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Cannot read image from {args.image}")
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Config
    if args.quick:
        config = quick_config()
    else:
        config = CASTConfig()

    config.device = args.device
    config.output_dir = args.output
    config.sam3d_model_id = args.sam3d_model_id
    config.sam3d_offline = args.sam3d_offline
    config.sam3d_use_fp16 = not args.sam3d_fp32
    config.pose_refinement = args.pose_refinement
    config.vlm_ensemble_trials = args.vlm_trials

    # Configure VLM
    config.vlm_provider = args.vlm
    if args.vlm == 'qwen':
        config.qwen_api_key = args.qwen_key
        config.qwen_base_url = args.qwen_base_url
        config.qwen_model = args.qwen_model
    elif args.vlm == 'openai':
        config.openai_api_key = args.openai_key
        config.openai_base_url = args.openai_base_url
        config.gpt_model = args.openai_model

    print(f"CAST Pipeline Configuration:")
    print(f"  Device:          {config.device}")
    print(f"  VLM provider:    {config.vlm_provider}")
    print(f"  VLM model:       {config.get_vlm_model()}")
    print(f"  VLM trials:      {config.vlm_ensemble_trials}")
    print(f"  SAM 3D offline:  {config.sam3d_offline}")
    print(f"  Pose refinement: {config.pose_refinement}")
    print(f"  Physics corr:    {config.enable_physics_correction}")

    # Run pipeline
    pipe = CASTPipeline(config)
    scene = pipe.reconstruct(img, output_dir=args.output)

    print(f"\nDone! Results saved to {args.output}/")


if __name__ == '__main__':
    main()
