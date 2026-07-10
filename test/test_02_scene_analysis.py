"""test_02: 场景分析 — CAST Stage 1 (Section 3)

对应 CAST 论文 Section 3 — Preprocessing:
  1. Florence-2 → 目标检测 + 描述 + 边界框
  2. GroundedSAMv2 → 精细化分割掩膜 {M_i}
  3. MoGe → 像素对齐深度图 + 相机内参
  4. 深度→点云 → 逐物体点云 {q_i}（场景坐标系）

用法:
  python test/test_02_scene_analysis.py --image data/images/room.png --output ./test/output
"""

import argparse, os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2, numpy as np
from config import CASTConfig

def main():
    p = argparse.ArgumentParser(description="Step 2: Scene Analysis")
    p.add_argument('--image', required=True)
    p.add_argument('--output', default='./test/output')
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  TEST 02: Scene Analysis (CAST Stage 1 / Section 3)")
    print("=" * 60)

    # Load image
    img_bgr = cv2.imread(args.image)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    print(f"  Image: {h} x {w}")

    # -------------------------------------------------------
    # Step 1: Object detection — Florence-2
    # -------------------------------------------------------
    print("\n--- 2.1 Object Detection (Florence-2) ---")
    from scene_analysis import detect_objects_florence2

    detections = detect_objects_florence2(img)
    if detections:
        print(f"  Detected {len(detections)} objects:")
        for d in detections:
            print(f"    - {d.get('name', 'unknown')}: {d.get('description','')} "
                  f"bbox={d.get('bbox')}")
    else:
        print("  [STUB] Florence-2 unavailable — treating whole image as 1 object")
        detections = [{'name': 'scene', 'description': 'full image', 'bbox': (0, 0, w, h)}]

    # -------------------------------------------------------
    # Step 2: Segmentation — GroundedSAMv2
    # -------------------------------------------------------
    print("\n--- 2.2 Segmentation (GroundedSAMv2) ---")
    from scene_analysis import segment_objects_grounded_sam

    bboxes = [d['bbox'] for d in detections]
    labels = [d.get('name', 'object') for d in detections]
    masks = segment_objects_grounded_sam(img, bboxes, labels)
    print(f"  Created {len(masks)} masks: sizes = {[m.sum() for m in masks]}")

    # -------------------------------------------------------
    # Step 3: Depth estimation — MoGe
    # -------------------------------------------------------
    print("\n--- 2.3 Depth Estimation (MoGe) ---")
    from scene_analysis import estimate_depth_moge

    depth_map, intrinsics = estimate_depth_moge(img)
    print(f"  Depth map:  {depth_map.shape}, range=[{depth_map.min():.2f},{depth_map.max():.2f}]")
    print(f"  Intrinsics:\n{intrinsics}")

    # -------------------------------------------------------
    # Step 4: Depth → Per-object point clouds
    # -------------------------------------------------------
    print("\n--- 2.4 Depth → Per-Object Point Clouds ---")
    from scene_analysis import depth_to_point_cloud, ObjectInfo

    # Compute occlusion mask per object
    occlusion_masks = []
    for i, m in enumerate(masks):
        occ = m.copy()
        for j, other in enumerate(masks):
            if j != i:
                occ[other > 0] = 0
        occlusion_masks.append(occ)

    objects = []
    for i, det in enumerate(detections):
        pc = depth_to_point_cloud(depth_map, intrinsics, masks[i])
        obj = ObjectInfo(
            id=i,
            name=det.get('name', f'object_{i}'),
            description=det.get('description', ''),
            bbox=det.get('bbox', (0, 0, w, h)),
            mask=masks[i],
            point_cloud=pc,
            occlusion_mask=occlusion_masks[i],
        )
        objects.append(obj)
        print(f"  [{i}] {obj.name}: {pc.shape[0]} points, "
              f"bbox={obj.bbox}, "
              f"visible_area={obj.occlusion_mask.sum()}/{occlusion_masks[i].sum()}")

    # -------------------------------------------------------
    # Save intermediate results
    # -------------------------------------------------------
    # Save per-object data (point clouds + masks as npz)
    for i, obj in enumerate(objects):
        np.savez_compressed(
            os.path.join(args.output, f'object_{i:02d}.npz'),
            id=i,
            name=obj.name,
            description=obj.description,
            bbox=np.array(obj.bbox),
            mask=obj.mask,
            point_cloud=obj.point_cloud,
            occlusion_mask=obj.occlusion_mask,
        )

    # Save scene-level data
    np.savez_compressed(
        os.path.join(args.output, 'scene_data.npz'),
        depth_map=depth_map,
        intrinsics=intrinsics,
        image=img,
        num_objects=len(objects),
    )

    # Save metadata
    meta = {
        'num_objects': len(objects),
        'object_names': [obj.name for obj in objects],
        'image_shape': [h, w],
        'depth_model': 'moge',
        'detection_model': 'florence-2',
        'segmentation_model': 'groundedsamv2',
    }
    with open(os.path.join(args.output, 'scene_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # ---- 自动生成各步骤可视化 ----
    from utils.visualization import visualize_all_steps
    visualize_all_steps(args.output, args.output)

    print(f"\n  [PASS] Stage 1 complete — {len(objects)} objects analyzed.")
    print(f"  Saved: {args.output}/object_*.npz, scene_data.npz, scene_meta.json")
    print(f"  Next:  python test/test_03_object_generation.py --output {args.output}")

if __name__ == '__main__':
    main()