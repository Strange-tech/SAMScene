"""test_06: 导出最终场景"""
import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, open3d as o3d

def main():
    p = argparse.ArgumentParser(description="Step 6: Export")
    p.add_argument('--output', default='./test/output')
    p.add_argument('--save-combined', action='store_true')
    args = p.parse_args()

    print("="*60)
    print("  TEST 06: Export Final Scene")
    print("="*60)

    with open(os.path.join(args.output, 'generation_results.json')) as f:
        gen_results = json.load(f)

    physics_file = os.path.join(args.output, 'physics_results.json')
    transforms = {}
    if os.path.exists(physics_file):
        with open(physics_file) as f:
            transforms = json.load(f)['transforms']
        print("  Using physics-corrected transforms")
    else:
        for r in gen_results:
            transforms[str(r['id'])] = {'R':r['R'],'t':r['t'],'s':r['s']}
        print("  Using pre-physics transforms")

    with open(os.path.join(args.output, 'relation_graph.json')) as f:
        relation_graph = json.load(f)

    export_dir = os.path.join(args.output, 'final_scene')
    os.makedirs(export_dir, exist_ok=True)
    scene_graph = {'objects':[], 'relation_graph':relation_graph}
    combined = o3d.geometry.TriangleMesh()

    for r in gen_results:
        oid = str(r['id'])
        mesh = o3d.io.read_triangle_mesh(r['mesh_path'])
        T = transforms.get(oid, {'R':r['R'],'t':r['t'],'s':r['s']})
        R = np.array(T['R']); t = np.array(T['t']); s = float(T['s'])

        mesh_f = o3d.geometry.TriangleMesh(mesh)
        mesh_f.scale(s, center=(0,0,0))
        mesh_f.rotate(R.T, center=(0,0,0))
        mesh_f.translate(t)

        obj_path = os.path.join(export_dir, f'{r["name"]}_{oid}.obj')
        o3d.io.write_triangle_mesh(obj_path, mesh_f)
        print(f"  [{oid}] {r['name']}: {len(mesh_f.vertices)} verts")

        scene_graph['objects'].append({
            'id':r['id'], 'name':r['name'],
            'mesh_file':f'{r["name"]}_{oid}.obj',
            'transform':{'R':T['R'],'t':T['t'],'s':T['s']},
        })
        combined += mesh_f

    if args.save_combined:
        cp = os.path.join(export_dir, 'combined_scene.obj')
        o3d.io.write_triangle_mesh(cp, combined)
        print(f"  Combined: {cp}")

    with open(os.path.join(export_dir, 'scene_graph.json'),'w') as f:
        json.dump(scene_graph, f, indent=2)

    print(f"\n  Total: {len(gen_results)} objects")
    print(f"  Contact edges: {len(relation_graph.get('contact_edges',[]))}")
    print(f"  Support edges: {len(relation_graph.get('support_edges',[]))}")
    print(f"  Output: {export_dir}/")
    print(f"\n  [PASS] All 6 stages complete!")

if __name__ == '__main__':
    main()