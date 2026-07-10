"""test_01: 加载并验证输入图像

对应 CAST 论文: 输入阶段
用法: python test/test_01_load_image.py --image data/images/room.png
"""

import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cv2, numpy as np

def main():
    p = argparse.ArgumentParser(description="Step 1: Load input image")
    p.add_argument('--image', required=True, help='Path to RGB image')
    p.add_argument('--output', default='./test/output')
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  TEST 01: Load & Validate Input Image")
    print("=" * 60)

    # 1. Read
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"[FAIL] Cannot read: {args.image}")
        sys.exit(1)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    print(f"  Source:  {args.image}")
    print(f"  Shape:   {h} x {w} x {c}")
    print(f"  Dtype:   {img.dtype}")
    print(f"  Range:   [{img.min()}, {img.max()}]")

    for i, ch in enumerate(['R','G','B']):
        arr = img[:,:,i]
        print(f"  {ch}: mean={arr.mean():.1f}  std={arr.std():.1f}")

    # 2. Save copy
    out_img = os.path.join(args.output, 'input_image.png')
    cv2.imwrite(out_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"\n  Saved: {out_img}")

    # 3. Save meta
    meta = {
        'image_path': os.path.abspath(args.image),
        'shape': [h, w, c],
    }
    with open(os.path.join(args.output, 'image_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  [PASS]  Step 1 complete.")
    print(f"  Next:   python test/test_02_scene_analysis.py --image {args.image} --output {args.output}")

if __name__ == '__main__':
    main()