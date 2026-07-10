"""test_04: 场景关系图 — CAST Stage 3 (Section 5.3)

对应 CAST 论文 Section 5.3 — Scene Relation Graph:
  1. Set-of-Mark (SoM): 用随机颜色+编号标注每个物体的mask
  2. VLM 查询: 发送 SoM 图像 → Qwen-VL / GPT-4V
  3. 解析: 提取 6 种细粒度关系 (Stack/Lean/Hang/Clamped/Contained/Edge-Point)
  4. 集成: 多次采样 → 多数投票 (≥半数)
  5. 映射: 细粒度关系 → Contact (双向) + Support (单向)

用法:
  python test/test_04_relation_graph.py --output ./test/output [--qwen-key sk-xxx]
"""

import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from config import CASTConfig

def main():
    p = argparse.ArgumentParser(description="Step 4: Scene Relation Graph via VLM")
    p.add_argument('--output', default='./test/output')
    p.add_argument('--qwen-key', default=None, help='DashScope API key')
    p.add_argument('--openai-key', default=None, help='OpenAI API key (fallback)')
    p.add_argument('--vlm', default='qwen', choices=['qwen','openai'])
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--save-som', action='store_true', help='Save SoM visualization')
    args = p.parse_args()

    print("=" * 60)
    print("  TEST 04: Scene Relation Graph")
    print("  (CAST Stage 3 / Section 5.3)")
    print("=" * 60)

    # Load scene data
    scene_data = np.load(os.path.join(args.output, 'scene_data.npz'))
    img = scene_data['image']
    num_objects = int(scene_data['num_objects'])
    print(f"  Image: {img.shape[:2]}, Objects: {num_objects}")

    if num_objects < 2:
        print("\n  [SKIP] Only 1 object — no relations needed.")
        graph = {'nodes': [0], 'contact_edges': [], 'support_edges': []}
        with open(os.path.join(args.output, 'relation_graph.json'), 'w') as f:
            json.dump(graph, f, indent=2)
        return

    # Collect objects with masks for SoM
    objects_for_som = []
    for i in range(num_objects):
        obj_data = np.load(os.path.join(args.output, f'object_{i:02d}.npz'))
        objects_for_som.append({
            'id': int(obj_data['id']),
            'name': str(obj_data['name']),
            'mask': obj_data['mask'],
        })

    # ---- Step 1: Render Set-of-Mark ----
    print(f"\n--- 4.1 Render Set-of-Mark ---")
    from utils.relation_graph import render_set_of_mark, encode_image_b64

    som_img = render_set_of_mark(img, objects_for_som, colorize=True)
    if args.save_som:
        som_path = os.path.join(args.output, 'som_visualization.png')
        cv2.imwrite(som_path, cv2.cvtColor(som_img, cv2.COLOR_RGB2BGR))
        print(f"  SoM saved: {som_path}")
    print(f"  SoM rendered: {som_img.shape}")

    # ---- Step 2: Determine VLM provider & API key ----
    print(f"\n--- 4.2 VLM Configuration ---")
    if args.vlm == 'qwen':
        api_key = args.qwen_key or os.environ.get('DASHSCOPE_API_KEY')
        base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        model = 'qwen-vl-max'
    else:
        api_key = args.openai_key or os.environ.get('OPENAI_API_KEY')
        base_url = args.openai_base_url or 'https://api.openai.com/v1'
        model = 'gpt-4-vision-preview'

    if not api_key:
        print(f"  [WARNING] No API key for VLM ({args.vlm}).")
        print(f"  Set --qwen-key or DASHSCOPE_API_KEY for Qwen, or")
        print(f"  --openai-key or OPENAI_API_KEY for GPT-4V.")
        print(f"  Returning empty relation graph.")
        graph = {'nodes': list(range(num_objects)), 'contact_edges': [], 'support_edges': []}
        with open(os.path.join(args.output, 'relation_graph.json'), 'w') as f:
            json.dump(graph, f, indent=2)
        return

    print(f"  Provider: {args.vlm}")
    print(f"  Model:    {model}")
    print(f"  Trials:   {args.trials} (majority vote threshold: {max(1, args.trials//2)})")

    # ---- Step 3: Query VLM with SoM ----
    print(f"\n--- 4.3 VLM Query + Ensemble ---")
    from utils.relation_graph import query_vlm_with_som

    graph = query_vlm_with_som(
        image=img,
        objects=objects_for_som,
        api_key=api_key,
        base_url=base_url,
        model=model,
        ensemble_trials=args.trials,
        verbose=True,
    )

    # ---- Step 4: Display results ----
    print(f"\n--- 4.4 Results ---")
    print(f"  Nodes:          {graph['nodes']}")
    print(f"  Contact edges:  {graph['contact_edges']}")
    print(f"  Support edges:  {graph['support_edges']}")

    # Map to human-readable names
    id2name = {int(obj_data['id']): str(obj_data['name'])
               for i in range(num_objects)
               for obj_data in [np.load(os.path.join(args.output, f'object_{i:02d}.npz'))]}

    if graph['contact_edges']:
        print("\n  Contact pairs (bidirectional):")
        for (i,j) in graph['contact_edges']:
            print(f"    {id2name.get(i,i)} <-> {id2name.get(j,j)}")
    if graph['support_edges']:
        print("\n  Support pairs (supporter → supported):")
        for (supp, supd) in graph['support_edges']:
            print(f"    {id2name.get(supp,supp)} → {id2name.get(supd,supd)}")

    # Save relation graph
    # Convert tuple keys to lists for JSON
    json_graph = {
        'nodes': graph['nodes'],
        'contact_edges': [list(e) for e in graph['contact_edges']],
        'support_edges': [list(e) for e in graph['support_edges']],
    }
    with open(os.path.join(args.output, 'relation_graph.json'), 'w') as f:
        json.dump(json_graph, f, indent=2)

    print(f"\n  [PASS] Stage 3 complete.")
    print(f"  Next:  python test/test_05_physics_correction.py --output {args.output}")

if __name__ == '__main__':
    main()