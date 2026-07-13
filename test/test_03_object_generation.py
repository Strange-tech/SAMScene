import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cv2, numpy as np, open3d as o3d

def merge_scene_obj(poses, output_dir):
    '''Merge individual meshes into scene OBJ using extracted poses.'''
    all_verts, all_faces, voff = [], [], 0
    print(f'\n--- Merging scene OBJ ({len(poses)} objects) ---')
    for p in poses:
        mp, name = p['mesh_path'], p['name']
        R = np.array(p['R'], dtype=np.float64)
        t = np.array(p['t'], dtype=np.float64)
        s = float(p['s'])
        if not os.path.exists(mp):
            print(f'  [SKIP] {name}: not found'); continue
        verts, faces = [], []
        with open(mp) as f:
            for line in f:
                if line.startswith('v '):
                    v = line.strip().split()
                    verts.append([float(v[1]), float(v[2]), float(v[3])])
                elif line.startswith('f '):
                    fc = line.strip().split()
                    faces.append([int(x.split('/')[0]) - 1 for x in fc[1:]])
        verts = np.array(verts, dtype=np.float64)
        verts = (s * (R @ verts.T)).T + t
        all_verts.append(verts)
        for face in faces:
            all_faces.append([idx + voff + 1 for idx in face])
        voff += len(verts)
        print(f'  [{p["id"]}] {name}: {len(verts):,} verts, {len(faces):,} faces [s={s:.2f}]')

    out = os.path.join(output_dir, 'scene_combined.obj')
    tv = sum(len(v) for v in all_verts); tf = len(all_faces)
    with open(out, 'w') as f:
        f.write(f'# Combined scene\n# {len(poses)} objects, {tv} verts, {tf} faces\n\n')
        for verts in all_verts:
            for v in verts:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in all_faces:
            f.write(f'f {face[0]} {face[1]} {face[2]}\n')
    sz = os.path.getsize(out) / (1024*1024)
    print(f'  => {out}\n     {tv:,} verts, {tf:,} faces, {sz:.1f} MB')


def main():
    p = argparse.ArgumentParser(description='Step 3: Object Generation')
    p.add_argument('--output', default='./test/output')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default='sam3d', choices=['midi', 'sam3d'])
    p.add_argument('--no-combined', action='store_true')
    args = p.parse_args()

    print('=' * 60)
    print(f'  TEST 03: 3D Object Generation (model={args.model.upper()})')
    print('=' * 60)

    sd = np.load(os.path.join(args.output, 'scene_data.npz'), allow_pickle=True)
    img = sd['image']
    n_obj = int(sd['num_objects'])
    print(f'  Loaded scene: {n_obj} objects, image={img.shape[:2]}')

    all_names, all_masks, all_bboxes = [], [], []
    for i in range(n_obj):
        d = np.load(os.path.join(args.output, f'object_{i:02d}.npz'), allow_pickle=True)
        all_names.append(str(d['name']))
        all_masks.append(d['mask'])
        all_bboxes.append(d['bbox'])

    if args.model == 'midi':
        from midi_wrapper import MIDIWrapper
        midi = MIDIWrapper(device=args.device)
        if not midi.is_available:
            print('\n  [FAIL] MIDI not available.'); return
        meshes, transforms = midi.generate_from_scene(
            image=img, masks=all_masks, bboxes=all_bboxes,
            seed=args.seed, num_inference_steps=50, guidance_scale=3.0,
        )
        results = []
        for i in range(n_obj):
            mesh = meshes[i]; T = transforms[i]; name = all_names[i]
            nv = len(mesh.vertices) if mesh.vertices else 0
            mp = os.path.join(args.output, f'mesh_{i:02d}_{name}.obj')
            if nv > 0: o3d.io.write_triangle_mesh(mp, mesh)
            print(f'  [{i}] {name}: {nv} verts -> {mp}')
            results.append({
                'id': i, 'name': name, 'mesh_path': mp,
                'R': T['R'].tolist(), 't': T['t'].tolist(), 's': float(T['s']),
            })
        if not args.no_combined:
            merge_scene_obj(results, args.output)

    else:
        from sam3d_wrapper import SAM3DWrapper
        sam3d = SAM3DWrapper(device=args.device)

        # ---- Official pipeline: Inference + make_scene → PLY ----
        ply_path = os.path.join(args.output, 'scene_gaussians.ply')
        _, meshes, poses = sam3d.generate_scene_ply(img, all_masks, ply_path, seed=args.seed)

        # ---- Save individual meshes ----
        results = []
        for i, (mesh, pose, name) in enumerate(zip(meshes, poses, all_names)):
            nv = len(mesh.vertices) if mesh.vertices else 0
            mp = os.path.join(args.output, f'mesh_{i:02d}_{name}.obj')
            if nv > 0: o3d.io.write_triangle_mesh(mp, mesh)
            print(f'  [{i}] {name}: {nv} verts -> {mp}')
            results.append({
                'id': i, 'name': name, 'mesh_path': mp,
                'R': pose['R'].tolist() if isinstance(pose['R'], np.ndarray) else pose['R'],
                't': pose['t'].tolist() if isinstance(pose['t'], np.ndarray) else pose['t'],
                's': float(pose['s']),
            })

        # ---- Merge OBJ (optional) ----
        if not args.no_combined:
            merge_scene_obj(results, args.output)

    with open(os.path.join(args.output, 'generation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n  [PASS] Stage 2 complete - {len(results)}/{n_obj} objects ({args.model.upper()}).')

if __name__ == '__main__':
    main()
