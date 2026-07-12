import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cv2, numpy as np, open3d as o3d, torch

def merge_scene_obj(results, output_dir):
    all_verts, all_faces, voff = [], [], 0
    print(f'\n--- Merging scene OBJ ({len(results)} objects) ---')
    for r in results:
        mp, name = r['mesh_path'], r['name']
        R = np.array(r['R'], dtype=np.float64)
        t = np.array(r['t'], dtype=np.float64)
        s = float(r['s'])
        if not os.path.exists(mp):
            print(f'  [SKIP] {name}: not found')
            continue
        verts, faces = [], []
        with open(mp) as f:
            for line in f:
                if line.startswith('v '):
                    p = line.strip().split()
                    verts.append([float(p[1]), float(p[2]), float(p[3])])
                elif line.startswith('f '):
                    p = line.strip().split()
                    faces.append([int(x.split('/')[0]) - 1 for x in p[1:]])
        verts = np.array(verts, dtype=np.float64)
        verts = (s * (R @ verts.T)).T + t
        all_verts.append(verts)
        for face in faces:
            all_faces.append([idx + voff + 1 for idx in face])
        voff += len(verts)
        tf_str = f' [s={s:.2f}, t=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})]'
        print(f'  [{r["id"]}] {name}: {len(verts):,} verts, {len(faces):,} faces{tf_str}')

    out = os.path.join(output_dir, 'scene_combined.obj')
    tv = sum(len(v) for v in all_verts)
    tf = len(all_faces)
    with open(out, 'w') as f:
        f.write(f'# Combined scene\n# {len(results)} objects, {tv} verts, {tf} faces\n\n')
        for verts in all_verts:
            for v in verts:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in all_faces:
            f.write(f'f {face[0]} {face[1]} {face[2]}\n')
    sz = os.path.getsize(out) / (1024*1024)
    print(f'  => {out}\n     {tv:,} verts, {tf:,} faces, {sz:.1f} MB')
    return out


def main():
    p = argparse.ArgumentParser(description='Step 3: Object Generation')
    p.add_argument('--output', default='./test/output')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--guidance-scale', type=float, default=3.0)
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

    all_names, all_masks = [], []
    for i in range(n_obj):
        d = np.load(os.path.join(args.output, f'object_{i:02d}.npz'), allow_pickle=True)
        all_names.append(str(d['name']))
        all_masks.append(d['mask'])

    results = []

    # ================================================================
    # MIDI
    # ================================================================
    if args.model == 'midi':
        from midi_wrapper import MIDIWrapper
        midi = MIDIWrapper(device=args.device)
        if not midi.is_available:
            print('\n  [FAIL] MIDI not available.')
            return

        bboxes = [d['bbox'] for d in ...]  # reload with bboxes
        all_bboxes = []
        for i in range(n_obj):
            d = np.load(os.path.join(args.output, f'object_{i:02d}.npz'), allow_pickle=True)
            all_bboxes.append(d['bbox'])

        meshes, transforms = midi.generate_from_scene(
            image=img, masks=all_masks, bboxes=all_bboxes,
            seed=args.seed, num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
        for i in range(n_obj):
            mesh = meshes[i]; T = transforms[i]; name = all_names[i]
            nv = len(mesh.vertices) if mesh.vertices else 0
            mp = os.path.join(args.output, f'mesh_{i:02d}_{name}.obj')
            if nv > 0: o3d.io.write_triangle_mesh(mp, mesh)
            print(f'  [{i}] {name}: {nv} verts -> {mp}')
            results.append({
                'id': i, 'name': name, 'mesh_path': mp,
                'R': T['R'].tolist() if isinstance(T['R'], np.ndarray) else T['R'],
                't': T['t'].tolist() if isinstance(T['t'], np.ndarray) else T['t'],
                's': float(T['s']), 'n_vertices': nv,
            })

    # ================================================================
    # SAM3D — full image + per-object mask (official multi-object API)
    # ================================================================
    else:
        from sam3d_wrapper import SAM3DWrapper
        sam3d = SAM3DWrapper(device=args.device)
        if not sam3d.is_available:
            print('\n  [FAIL] SAM3D not available.')
            return

        print(f'\n  [SAM3D] Generating {n_obj} objects (full image + per-object mask) ...')
        for i in range(n_obj):
            name = all_names[i]
            mask = all_masks[i]

            print(f'\n  [{i}] {name}: mask={mask.shape}, nonzero={mask.sum()}')

            try:
                torch.cuda.empty_cache()
                mesh, R, t, s = sam3d.generate(
                    image=img,     # FULL image, NOT cropped
                    mask=mask,     # per-object mask
                )

                nv = len(mesh.vertices) if mesh.vertices else 0
                print(f'    mesh: {nv} verts')
                print(f'    pose:  s={s:.3f}, t=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})')

                mp = os.path.join(args.output, f'mesh_{i:02d}_{name}.obj')
                if nv > 0:
                    o3d.io.write_triangle_mesh(mp, mesh)
                print(f'  [{i}] {name}: {nv} verts -> {mp}')

                results.append({
                    'id': i, 'name': name, 'mesh_path': mp,
                    'R': R.tolist() if isinstance(R, np.ndarray) else R,
                    't': t.tolist() if isinstance(t, np.ndarray) else t,
                    's': float(s),
                    'n_vertices': nv,
                })
            except Exception as e:
                print(f'  [SKIP] {name}: {str(e)[:150]}')

    # ---- save & merge ----
    with open(os.path.join(args.output, 'generation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if not args.no_combined and results:
        merge_scene_obj(results, args.output)

    print(f'\n  [PASS] Stage 2 complete - {len(results)}/{n_obj} objects ({args.model.upper()}).')
    print(f'  Next:  python test/test_04_relation_graph.py --output {args.output}')

if __name__ == '__main__':
    main()
