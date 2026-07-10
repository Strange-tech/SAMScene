"""test_05: 物理感知校正 — CAST Stage 4 (Section 5)

对应 CAST 论文 Section 5 — Physics-Aware Correction:
  1. 构建约束图 (来自 Stage 3 的关系图)
  2. 表面点采样 (每物体 2048 点)
  3. SDF 计算 (Open3D RaycastingScene)
  4. 优化 (Adam over 6D rotation + 3D translation):
     - Contact Cost (Eq.9): 双向, 穿透惩罚 + 最小距离惩罚
     - Support Cost (Eq.10): 单向, 仅优化被支撑物体
     - 近表面正则化 (Eq.11): sigma 带内鼓励紧密接触

用法:
  python test/test_05_physics_correction.py --output ./test/output
"""

import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import open3d as o3d
import torch
from config import CASTConfig

def main():
    p = argparse.ArgumentParser(description="Step 5: Physics-Aware Correction")
    p.add_argument('--output', default='./test/output')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--sigma', type=float, default=0.05,
                   help='Near-surface threshold (paper Eq.11)')
    args = p.parse_args()

    print("=" * 60)
    print("  TEST 05: Physics-Aware Correction")
    print("  (CAST Stage 4 / Section 5)")
    print("=" * 60)

    # Load relation graph from Stage 3
    with open(os.path.join(args.output, 'relation_graph.json')) as f:
        graph = json.load(f)
    # Convert back to tuples
    graph['contact_edges'] = [tuple(e) for e in graph['contact_edges']]
    graph['support_edges'] = [tuple(e) for e in graph['support_edges']]

    print(f"  Contact edges:  {graph['contact_edges']}")
    print(f"  Support edges:  {graph['support_edges']}")

    if not graph['contact_edges'] and not graph['support_edges']:
        print("\n  [SKIP] No relation edges — physics correction not needed.")
        results = {}
        for i in graph['nodes']:
            results[str(i)] = {'R': np.eye(3).tolist(), 't': [0,0,0], 's': 1.0}
        with open(os.path.join(args.output, 'physics_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        return

    # Load meshes from Stage 2
    with open(os.path.join(args.output, 'generation_results.json')) as f:
        gen_results = json.load(f)

    meshes = []
    objects_for_physics = []
    for r in gen_results:
        mesh = o3d.io.read_triangle_mesh(r['mesh_path'])
        mesh.compute_vertex_normals()
        meshes.append(mesh)
        # Create a minimal object-like struct
        from dataclasses import dataclass
        @dataclass
        class ObjInfo:
            id: int; point_cloud: np.ndarray
        obj_data = np.load(os.path.join(args.output, f'object_{r["id"]:02d}.npz'))
        objects_for_physics.append(ObjInfo(
            id=r['id'],
            point_cloud=obj_data['point_cloud'],
        ))
        print(f"  Loaded [{r['id']}] {r['name']}: "
              f"{len(mesh.vertices)} verts, {len(mesh.triangles)} faces")

    # ---- Step 1: Pre-sample surface points ----
    print(f"\n--- 5.1 Surface Sampling (2048 pts/object) ---")
    from physics_correction import sample_surface_points

    surface_samples = {}
    for i, mesh in enumerate(meshes):
        pts = sample_surface_points(mesh, 2048)
        surface_samples[i] = torch.from_numpy(pts).float()
        print(f"  [{i}] {pts.shape[0]} points")

    # ---- Step 2: Build constraint graph ----
    print(f"\n--- 5.2 Build Constraint Graph ---")
    from physics_correction import build_constraint_graph

    constraints = build_constraint_graph(graph, objects_for_physics, meshes, surface_samples)
    print(f"  Variables: {len(constraints['variables'])} objects")
    print(f"  Contact constraints: {len(constraints['contacts'])}")
    print(f"  Support constraints: {len(constraints['supports'])}")

    # ---- Step 3: Setup optimizer ----
    print(f"\n--- 5.3 Optimization ({args.steps} steps, lr={args.lr}) ---")

    params = []
    for v in constraints['variables'].values():
        params.append(v['R_6d'])
        params.append(v['t'])
    optim = torch.optim.Adam(params, lr=args.lr)

    # ---- Step 4: Optimization loop (Eq.8) ----
    loss_history = []
    for step in range(args.steps):
        optim.zero_grad()
        total_loss = torch.tensor(0.0)

        for (i, j, cost_fn) in constraints['contacts']:
            total_loss = total_loss + cost_fn(constraints['variables'])

        for (supp, supd, cost_fn) in constraints['supports']:
            total_loss = total_loss + cost_fn(constraints['variables'])

        if torch.is_tensor(total_loss) and total_loss > 0:
            total_loss.backward()
            optim.step()

        loss_val = total_loss.item() if torch.is_tensor(total_loss) else 0.0
        loss_history.append(loss_val)

        if step % 50 == 0:
            print(f"  step {step:4d}/{args.steps}  loss={loss_val:.6f}")

    print(f"  Final loss: {loss_history[-1]:.6f}")

    # ---- Step 5: Extract optimized transforms ----
    print(f"\n--- 5.4 Extract Optimized Transforms ---")
    from physics_correction import rotation_6d_to_matrix

    results = {}
    for obj_id, var in constraints['variables'].items():
        R = rotation_6d_to_matrix(var['R_6d']).detach().cpu().numpy()
        t = var['t'].detach().cpu().numpy()
        results[str(obj_id)] = {
            'R': R.tolist(),
            't': t.tolist(),
            's': 1.0,
        }

        # Compare to original
        orig_R = np.array(gen_results[obj_id]['R'])
        orig_t = np.array(gen_results[obj_id]['t'])
        delta_t = np.linalg.norm(t - orig_t)
        print(f"  [{obj_id}] delta_t={delta_t:.4f}m")

    # Save results
    physics_results = {
        'transforms': results,
        'loss_history': [float(v) for v in loss_history],
        'final_loss': float(loss_history[-1]) if loss_history else 0,
        'config': {
            'steps': args.steps,
            'lr': args.lr,
            'sigma': args.sigma,
        },
    }
    with open(os.path.join(args.output, 'physics_results.json'), 'w') as f:
        json.dump(physics_results, f, indent=2)

    print(f"\n  [PASS] Stage 4 complete.")
    print(f"  Next:  python test/test_06_export.py --output {args.output}")

if __name__ == '__main__':
    main()