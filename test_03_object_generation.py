"""test_03: 多实例 3D 物体生成 — CAST Stage 2 (Section 4)

使用 MIDI (Multi-Instance Diffusion) 替代原 SAM 3D:
  - MIDI 从完整场景图像 + 所有物体 mask 同时生成 3D mesh
  - 物体已在统一的场景坐标系中，无需 PoseAdapter 桥接
  - 替代了原 SAM 3D 逐物体生成 + PoseAdapter 的两步流程

用法:
  python test/test_03_object_generation.py --output ./test/output
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import open3d as o3d


def main():
    p = argparse.ArgumentParser(description="Step 3: Object Generation (MIDI)")
    p.add_argument('--output', default='./test/output')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=50,
                   help='MIDI 去噪步数 (默认 50)')
    p.add_argument('--guidance-scale', type=float, default=7.0,
                   help='CFG scale (默认 7.0)')
    args = p.parse_args()

    print("=" * 60)
    print("  TEST 03: Multi-Instance 3D Object Generation")
    print("  (CAST Stage 2 — MIDI 替代 SAM 3D)")
    print("=" * 60)

    # -------------------------------------------------------
    # 加载 test_02 输出数据
    # -------------------------------------------------------
    scene_data = np.load(os.path.join(args.output, 'scene_data.npz'),
                         allow_pickle=True)
    img = scene_data['image']
    num_objects = int(scene_data['num_objects'])
    print(f"  Loaded scene: {num_objects} objects, image={img.shape[:2]}")

    # 收集所有物体的 mask 和 bbox
    all_masks = []
    all_names = []
    all_bboxes = []
    for i in range(num_objects):
        obj_data = np.load(os.path.join(args.output, f'object_{i:02d}.npz'),
                           allow_pickle=True)
        all_names.append(str(obj_data['name']))
        all_masks.append(obj_data['mask'])
        all_bboxes.append(obj_data['bbox'])

    # -------------------------------------------------------
    # 初始化 MIDI
    # -------------------------------------------------------
    from midi_wrapper import MIDIWrapper

    midi = MIDIWrapper(device=args.device)
    print(f"  MIDI available: {midi.is_available}")

    if not midi.is_available:
        print("\n  [FAIL] MIDI 模型不可用，测试终止。")
        return

    # -------------------------------------------------------
    # MIDI 一次性生成所有物体
    # -------------------------------------------------------
    print(f"\n  [MIDI] 同时生成 {num_objects} 个物体 ...")
    meshes, transforms = midi.generate_from_scene(
        image=img,
        masks=all_masks,
        bboxes=all_bboxes,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    # -------------------------------------------------------
    # 保存结果
    # -------------------------------------------------------
    results = []
    for i in range(num_objects):
        mesh = meshes[i]
        T = transforms[i]
        name = all_names[i]

        n_verts = len(mesh.vertices) if mesh.vertices else 0
        n_faces = len(mesh.triangles) if mesh.triangles else 0

        mesh_path = os.path.join(args.output, f'mesh_{i:02d}_{name}.obj')
        if n_verts > 0:
            o3d.io.write_triangle_mesh(mesh_path, mesh)

        print(f"  [{i}] {name}: {n_verts} verts, {n_faces} faces -> {mesh_path}")

        results.append({
            'id': i,
            'name': name,
            'mesh_path': mesh_path,
            'R': T['R'],
            't': T['t'],
            's': float(T['s']),
            'n_vertices': n_verts,
        })

    # 保存 generation_results.json (兼容下游 test_04/05/06)
    with open(os.path.join(args.output, 'generation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  [PASS] Stage 2 complete — {len(results)} objects generated (MIDI).")
    print(f"  Next:  python test/test_04_relation_graph.py --output {args.output}")


if __name__ == '__main__':
    main()
