#!/usr/bin/env python3
"""
可视化 test_02 各步骤结果，输出到 output 目录:

  Step 2 — Segmentation (GroundedSAMv2): 分割 mask 叠加图
  Step 3 — Depth Estimation (MoGe): 深度图彩色可视化
  Step 4 — Point Clouds: 3D 点云多视图投影图

用法:
  python visualize_steps_all.py --output ./test/output
"""

import argparse, os, sys, json
import cv2, numpy as np

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128), (255, 128, 0), (0, 128, 128), (128, 128, 0),
]


def visualize_segmentation(input_dir: str, output_dir: str):
    """Step 2: 绘制 SAM 分割 mask + bbox 叠加图"""
    print("\n" + "=" * 60)
    print("  Step 2: Segmentation (GroundedSAMv2) 可视化")
    print("=" * 60)

    img = cv2.imread(os.path.join(input_dir, "input_image.png"))
    h, w = img.shape[:2]

    with open(os.path.join(input_dir, "scene_meta.json")) as f:
        meta = json.load(f)
    num = meta["num_objects"]

    # 创建 mask 叠加图层
    overlay = img.copy()

    for i in range(num):
        data = np.load(os.path.join(input_dir, f"object_{i:02d}.npz"), allow_pickle=True)
        name, bbox, mask = str(data["name"]), data["bbox"], data["mask"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = COLORS[i % len(COLORS)]

        # 半透明 mask 填充
        alpha = 0.4
        overlay[mask > 0] = (
            alpha * np.array(color) + (1 - alpha) * overlay[mask > 0]
        ).astype(np.uint8)

        # bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"[{i}] {name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        mask_px = int(mask.sum())
        print(f"  [{i}] {name:<16} mask={mask_px:>7} px  bbox=[{x1},{y1},{x2},{y2}]")

    # 并排显示: 原图 + mask叠加 (bbox) | mask 叠加 (半透明)
    result = np.hstack([img, overlay])
    cv2.putText(result, "Segmentation (bbox + labels)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "Segmentation (mask overlay)",
                (w + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    path = os.path.join(output_dir, "step2_segmentation.png")
    cv2.imwrite(path, result)
    print(f"\n  [OK] {path}")
    return path


def visualize_depth(input_dir: str, output_dir: str):
    """Step 3: 深度图彩色可视化"""
    print("\n" + "=" * 60)
    print("  Step 3: Depth Estimation (MoGe) 可视化")
    print("=" * 60)

    img = cv2.imread(os.path.join(input_dir, "input_image.png"))
    scene = np.load(os.path.join(input_dir, "scene_data.npz"), allow_pickle=True)
    depth = scene["depth_map"]

    d_min, d_max = depth.min(), depth.max()
    print(f"  Depth range: [{d_min:.2f}, {d_max:.2f}] m")

    # 归一化 → 彩色映射
    depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0, 1)
    depth_color = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )

    # 添加 colorbar
    bar_h = 20
    bar = np.zeros((bar_h, depth.shape[1], 3), dtype=np.uint8)
    for x in range(depth.shape[1]):
        v = x / depth.shape[1]
        bar[:, x] = cv2.applyColorMap(
            np.array([[int(v * 255)]], dtype=np.uint8), cv2.COLORMAP_TURBO
        )[0, 0]

    # 组合: 原图 | 深度图
    depth_resized = cv2.resize(depth_color, (img.shape[1], img.shape[0]))
    result = np.hstack([img, depth_resized])

    # 添加 colorbar 到底部
    result = cv2.copyMakeBorder(result, 0, bar_h + 30, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(30, 30, 30))
    bh = result.shape[0]
    result[bh - bar_h - 25:bh - 25, :bar.shape[1]] = bar

    # 标签
    cv2.putText(result, "RGB Input", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "MoGe Depth Map", (img.shape[1] + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"{d_min:.2f}m", (5, bh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, f"{d_max:.2f}m",
                (depth.shape[1] - 50, bh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    path = os.path.join(output_dir, "step3_depth_map.png")
    cv2.imwrite(path, result)
    print(f"  [OK] {path}")
    return path


def visualize_point_clouds(input_dir: str, output_dir: str):
    """Step 4: 3D 点云可视化（2D 投影图）"""
    print("\n" + "=" * 60)
    print("  Step 4: Per-Object Point Clouds 可视化")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    with open(os.path.join(input_dir, "scene_meta.json")) as f:
        meta = json.load(f)
    num = meta["num_objects"]

    colors_mpl = ['tab:green', 'tab:blue', 'tab:red', 'tab:cyan',
                  'tab:purple', 'tab:orange', 'tab:pink', 'tab:brown']

    # 收集所有点云
    all_objects = []
    for i in range(num):
        data = np.load(os.path.join(input_dir, f"object_{i:02d}.npz"), allow_pickle=True)
        name, pc = str(data["name"]), data["point_cloud"]
        all_objects.append({"name": name, "pc": pc, "id": i})
        print(f"  [{i}] {name:<16} {pc.shape[0]:>7} points  "
              f"X=[{pc[:,0].min():.2f},{pc[:,0].max():.2f}] "
              f"Y=[{pc[:,1].min():.2f},{pc[:,1].max():.2f}] "
              f"Z=[{pc[:,2].min():.2f},{pc[:,2].max():.2f}]")

    # 创建四视图: 3D全景 + 三个投影面
    fig = plt.figure(figsize=(20, 16))

    # --- 俯视图 (Top: X-Y) ---
    ax_t = fig.add_subplot(2, 2, 1)
    for obj, cm in zip(all_objects, colors_mpl):
        pc = obj["pc"]
        # 下采样以提高速度
        idx = np.random.choice(len(pc), min(2000, len(pc)), replace=False)
        ax_t.scatter(pc[idx, 0], pc[idx, 1], s=0.5, c=cm,
                     label=f'[{obj["id"]}] {obj["name"]}', alpha=0.7)
    ax_t.set_xlabel("X (m)"); ax_t.set_ylabel("Y (m)")
    ax_t.set_title("Top View (X-Y plane)", fontsize=13, fontweight='bold')
    ax_t.axis("equal")
    ax_t.legend(loc='upper right', fontsize=7, markerscale=5)

    # --- 前视图 (Front: X-Z) ---
    ax_f = fig.add_subplot(2, 2, 2)
    for obj, cm in zip(all_objects, colors_mpl):
        pc = obj["pc"]
        idx = np.random.choice(len(pc), min(2000, len(pc)), replace=False)
        ax_f.scatter(pc[idx, 0], pc[idx, 2], s=0.5, c=cm,
                     label=f'[{obj["id"]}] {obj["name"]}', alpha=0.7)
    ax_f.set_xlabel("X (m)"); ax_f.set_ylabel("Z (depth, m)")
    ax_f.set_title("Front View (X-Z plane)", fontsize=13, fontweight='bold')
    ax_f.legend(loc='upper right', fontsize=7, markerscale=5)

    # --- 侧视图 (Side: Y-Z) ---
    ax_s = fig.add_subplot(2, 2, 3)
    for obj, cm in zip(all_objects, colors_mpl):
        pc = obj["pc"]
        idx = np.random.choice(len(pc), min(2000, len(pc)), replace=False)
        ax_s.scatter(pc[idx, 1], pc[idx, 2], s=0.5, c=cm,
                     label=f'[{obj["id"]}] {obj["name"]}', alpha=0.7)
    ax_s.set_xlabel("Y (m)"); ax_s.set_ylabel("Z (depth, m)")
    ax_s.set_title("Side View (Y-Z plane)", fontsize=13, fontweight='bold')
    ax_s.legend(loc='upper right', fontsize=7, markerscale=5)

    # --- 3D 视图 ---
    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    for obj, cm in zip(all_objects, colors_mpl):
        pc = obj["pc"]
        idx = np.random.choice(len(pc), min(1500, len(pc)), replace=False)
        ax_3d.scatter(pc[idx, 0], pc[idx, 1], pc[idx, 2],
                      s=0.5, c=cm, label=f'[{obj["id"]}] {obj["name"]}', alpha=0.7)
    ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")
    ax_3d.set_title("3D Point Cloud", fontsize=13, fontweight='bold')
    ax_3d.legend(loc='upper right', fontsize=7, markerscale=5)
    ax_3d.view_init(elev=25, azim=-60)

    plt.suptitle("Per-Object Point Clouds (MoGe Depth → Camera Space)",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    path = os.path.join(output_dir, "step4_point_clouds.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  [OK] {path}")
    return path


def main():
    p = argparse.ArgumentParser(description="可视化 test_02 各步骤结果")
    p.add_argument("--output", default="./test/output",
                   help="输出目录（同时从此目录读取数据）")
    args = p.parse_args()

    print("=" * 60)
    print("  TEST 02 各步骤结果可视化")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)

    visualize_segmentation(args.output, args.output)
    visualize_depth(args.output, args.output)
    visualize_point_clouds(args.output, args.output)

    print(f"\n{'='*60}")
    print(f"  所有可视化结果已保存到: {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
