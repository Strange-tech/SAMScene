"""test_03: 感知3D实例生成 — CAST Stage 2 (Section 4)

对应 CAST 论文 Section 4 — Pose-Aware 3D Object Generation:
  Step 1 - SAM 3D: 单图 + mask → mesh + texture + (R,t,s)_cam
  Step 2 - PoseAdapter: 相机坐标系 → 场景坐标系
  替代了原论文的 ObjectGen + AlignGen 迭代循环

用法:
  python test/test_03_object_generation.py --output ./test/output
"""

import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import open3d as o3d
import cv2
from config import CASTConfig

def main():
    p = argparse.ArgumentParser(description="Step 3: Object Generation + Pose")
    p.add_argument('--output', default='./output')
    p.add_argument('--pose-refinement', default='icp',
                   choices=['none','umeyama','icp'])
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    print("=" * 60)
    print("  TEST 03: Pose-Aware 3D Object Generation")
    print("  (CAST Stage 2 / Section 4)")
    print("=" * 60)

    # Load scene analysis results
    scene_data = np.load(os.path.join(args.output, 'scene_data.npz'))
    img = scene_data['image']
    depth_map = scene_data['depth_map']
    intrinsics = scene_data['intrinsics']
    num_objects = int(scene_data['num_objects'])
    print(f"  Loaded scene: {num_objects} objects, image={img.shape[:2]}")

    # Init SAM 3D & PoseAdapter
    config = CASTConfig(
        device=args.device,
        pose_refinement=args.pose_refinement,
        sam3d_offline=False,
    )

    from sam3d_wrapper import SAM3DWrapper
    from pose_adapter import PoseAdapter

    sam3d = SAM3DWrapper(
        model_id=config.sam3d_model_id,
        device=args.device,
        use_fp16=config.sam3d_use_fp16,
        offline=config.sam3d_offline,
    )

    adapter = PoseAdapter(
        refinement=config.pose_refinement,
        icp_max_distance=config.icp_max_distance,
        verbose=True,
    )

    print(f"  SAM 3D available: {sam3d.is_available}")
    print(f"  Pose refinement:   {config.pose_refinement}")

    # Process each object
    results = []
    for i in range(num_objects):
        obj_data = np.load(os.path.join(args.output, f'object_{i:02d}.npz'))
        name = str(obj_data['name'])
        bbox = obj_data['bbox']
        mask = obj_data['mask']
        scene_pc = obj_data['point_cloud']
        occ_mask = obj_data['occlusion_mask']

        print(f"\n--- Object [{i}] {name} ---")
        print(f"  Scene PC points: {len(scene_pc)}")
        print(f"  Mask area:       {mask.sum()} pixels")

        # Crop image to bounding box
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(img.shape[1],x2), min(img.shape[0],y2)
        crop = img[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        print(f"  Crop: {x1},{y1} → {x2},{y2} (size={crop.shape[:2]})")

        # ---- Step 1: SAM 3D inference ----
        print(f"  [Step 1] SAM 3D generation ...")
        mesh, R_cam, t_cam, s_cam = sam3d.generate(
            image=crop,
            mask=mask_crop,
        )
        n_verts = len(mesh.vertices)
        print(f"    Mesh: {n_verts} verts, {len(mesh.triangles)} faces")
        print(f"    Pose (cam): s={s_cam:.3f}, t=({t_cam[0]:.3f},{t_cam[1]:.3f},{t_cam[2]:.3f})")

        # ---- Step 2: PoseAdapter (camera → scene) ----
        print(f"  [Step 2] PoseAdapter ({config.pose_refinement}) ...")
        if len(scene_pc) > 10:
            R_scene, t_scene, s_scene = adapter.adapt(
                mesh=mesh,
                sam3d_R=R_cam,
                sam3d_t=t_cam,
                sam3d_s=s_cam,
                scene_point_cloud=scene_pc,
                object_mask=mask,
            )
        else:
            print("    Too few scene points — passing through camera pose")
            R_scene, t_scene, s_scene = R_cam, t_cam, s_cam

        print(f"    Pose (scene): s={s_scene:.3f}, "
              f"t=({t_scene[0]:.3f},{t_scene[1]:.3f},{t_scene[2]:.3f})")

        # Save mesh
        mesh_path = os.path.join(args.output, f'mesh_{i:02d}_{name}.obj')
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"    Mesh saved: {mesh_path}")

        results.append({
            'id': i,
            'name': name,
            'mesh_path': mesh_path,
            'R': R_scene.tolist(),
            't': t_scene.tolist(),
            's': float(s_scene),
            'n_vertices': n_verts,
        })

    # Save generation results
    with open(os.path.join(args.output, 'generation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  [PASS] Stage 2 complete — {len(results)} objects generated.")
    print(f"  Next:  python test/test_04_relation_graph.py --output {args.output}")

if __name__ == '__main__':
    main()