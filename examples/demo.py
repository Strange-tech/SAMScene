"""
Minimal demo of the CAST pipeline.

Usage:
    python examples/demo.py --image path/to/image.jpg

Notes:
  - SAM 3D Objects (facebook/sam-3d-objects) is downloaded automatically
    from HuggingFace on first run (~7 GB). Set --sam3d-offline to skip.
  - Without a GPT-4V API key, the relation graph is empty and physics
    correction is skipped.
  - See README.md for setup instructions.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
from config import CASTConfig, quick_config
from pipeline import CASTPipeline


def main():
    parser = argparse.ArgumentParser(
        description="CAST: Single-Image 3D Scene Reconstruction (SAM 3D backend)")
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input RGB image')
    parser.add_argument('--output', type=str, default='./output/demo',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Use quick (lower-quality) config')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda / cpu)')

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

    # GPT-4V options
    parser.add_argument('--openai-key', type=str, default=None,
                        help='OpenAI API key for GPT-4V relation reasoning')
    parser.add_argument('--openai-base-url', type=str, default=None,
                        help='OpenAI API base URL (for proxies)')
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

    if args.openai_key:
        config.openai_api_key = args.openai_key
    if args.openai_base_url:
        config.openai_base_url = args.openai_base_url

    # Run pipeline
    pipe = CASTPipeline(config)
    scene = pipe.reconstruct(img, output_dir=args.output)

    print(f"\nDone! Results saved to {args.output}/")


if __name__ == '__main__':
    main()
