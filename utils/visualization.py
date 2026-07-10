"""
可视化工具集 — test_02 各步骤结果的可视化输出。

每个函数接受输入/输出目录，自动读取对应数据并生成可视化图像。

模块函数:
    visualize_detections()    — Florence-2 检测框 + 文本标签
    visualize_segmentation()  — SAM 分割 mask 叠加
    visualize_depth()         — MoGe 深度图彩色映射
    visualize_point_clouds()  — 逐物体点云四视图
    visualize_all_steps()     — 一次运行全部上述可视化
"""

import json
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 共享调色板
# ---------------------------------------------------------------------------

COLORS_BGR = [
    (0, 255, 0),     # 绿
    (255, 0, 0),     # 蓝
    (0, 0, 255),     # 红
    (255, 255, 0),   # 青
    (255, 0, 255),   # 品红
    (0, 255, 255),   # 黄
    (128, 0, 128),   # 紫
    (255, 128, 0),   # 橙
    (0, 128, 128),   # 深青
    (128, 128, 0),   # 橄榄
]

COLORS_MPL = [
    'tab:green', 'tab:blue', 'tab:red', 'tab:cyan',
    'tab:purple', 'tab:orange', 'tab:pink', 'tab:brown',
]


# ===================================================================
# Step 1: Florence-2 目标检测可视化
# ===================================================================

def visualize_detections(input_dir: str, output_dir: str) -> str:
    """Florence-2 检测框 + 标签，叠加到输入图像上。

    Args:
        input_dir:  包含 input_image.png, scene_meta.json, object_*.npz 的目录
        output_dir: 输出图像保存目录

    Returns:
        输出图像的绝对路径
    """
    img_path = os.path.join(input_dir, "input_image.png")
    if not os.path.exists(img_path):
        print(f"  [visualize_detections] 输入图像不存在: {img_path}")
        return ""

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 物体数量
    meta_path = os.path.join(input_dir, "scene_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            num = json.load(f).get("num_objects", 0)
    else:
        import glob
        num = len(sorted(glob.glob(os.path.join(input_dir, "object_*.npz"))))

    for i in range(num):
        npz_path = os.path.join(input_dir, f"object_{i:02d}.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path, allow_pickle=True)
        name = str(data["name"])
        bbox = data["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = COLORS_BGR[i % len(COLORS_BGR)]

        # 边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color,
                      max(2, min(w, h) // 400))

        # 标签
        label = f"[{i}] {name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = y1 - 5
        if label_y - th < 0:
            label_y = y1 + th + 5
        cv2.rectangle(img, (x1, label_y - th - 4),
                      (x1 + tw + 6, label_y + 4), color, -1)
        cv2.putText(img, label, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # 标题
    cv2.putText(img, "Florence-2 Object Detection",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "step1_florence2_detection.png")
    cv2.imwrite(out, img)
    print(f"  [vis] 检测可视化 -> {out}")
    return out


# ===================================================================
# Step 2: SAM 分割可视化
# ===================================================================

def visualize_segmentation(input_dir: str, output_dir: str) -> str:
    """SAM 分割 mask 半透明叠加 + bbox 标注。

    Args:
        input_dir:  包含 input_image.png, scene_meta.json, object_*.npz 的目录
        output_dir: 输出图像保存目录

    Returns:
        输出图像的绝对路径
    """
    img = cv2.imread(os.path.join(input_dir, "input_image.png"))
    h, w = img.shape[:2]

    with open(os.path.join(input_dir, "scene_meta.json")) as f:
        num = json.load(f)["num_objects"]

    overlay = img.copy()

    for i in range(num):
        data = np.load(os.path.join(input_dir, f"object_{i:02d}.npz"), allow_pickle=True)
        name, bbox, mask = str(data["name"]), data["bbox"], data["mask"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = COLORS_BGR[i % len(COLORS_BGR)]

        # 半透明 mask
        alpha = 0.4
        overlay[mask > 0] = (
            alpha * np.array(color) + (1 - alpha) * overlay[mask > 0]
        ).astype(np.uint8)

        # bbox + 标签
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"[{i}] {name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    result = np.hstack([img, overlay])
    cv2.putText(result, "Segmentation (bbox)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "Segmentation (mask)", (w + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "step2_segmentation.png")
    cv2.imwrite(out, result)
    print(f"  [vis] 分割可视化 -> {out}")
    return out


# ===================================================================
# Step 3: 深度图可视化
# ===================================================================

def visualize_depth(input_dir: str, output_dir: str) -> str:
    """MoGe 深度图 TURBO 彩色映射可视化。

    Args:
        input_dir:  包含 input_image.png, scene_data.npz 的目录
        output_dir: 输出图像保存目录

    Returns:
        输出图像的绝对路径
    """
    img = cv2.imread(os.path.join(input_dir, "input_image.png"))
    scene = np.load(os.path.join(input_dir, "scene_data.npz"), allow_pickle=True)
    depth = scene["depth_map"]

    d_min, d_max = depth.min(), depth.max()

    # TURBO colormap
    depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0, 1)
    depth_color = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    # colorbar
    bar_h = 20
    bar = np.zeros((bar_h, depth.shape[1], 3), dtype=np.uint8)
    for x in range(depth.shape[1]):
        bar[:, x] = cv2.applyColorMap(
            np.array([[int(x / depth.shape[1] * 255)]], dtype=np.uint8),
            cv2.COLORMAP_TURBO)[0, 0]

    depth_resized = cv2.resize(depth_color, (img.shape[1], img.shape[0]))
    result = np.hstack([img, depth_resized])
    result = cv2.copyMakeBorder(result, 0, bar_h + 30, 0, 0,
                                cv2.BORDER_CONSTANT, value=(30, 30, 30))
    bh = result.shape[0]
    result[bh - bar_h - 25:bh - 25, :bar.shape[1]] = bar

    cv2.putText(result, "RGB Input", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "MoGe Depth Map", (img.shape[1] + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"{d_min:.2f}m", (5, bh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, f"{d_max:.2f}m", (depth.shape[1] - 50, bh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "step3_depth_map.png")
    cv2.imwrite(out, result)
    print(f"  [vis] 深度图可视化 -> {out}")
    return out


# ===================================================================
# Step 4: 点云可视化
# ===================================================================

def visualize_point_clouds(input_dir: str, output_dir: str) -> str:
    """逐物体点云四视图 (Top / Front / Side / 3D)。

    Args:
        input_dir:  包含 scene_meta.json, object_*.npz 的目录
        output_dir: 输出图像保存目录

    Returns:
        输出图像的绝对路径
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(os.path.join(input_dir, "scene_meta.json")) as f:
        num = json.load(f)["num_objects"]

    all_objects = []
    for i in range(num):
        data = np.load(os.path.join(input_dir, f"object_{i:02d}.npz"), allow_pickle=True)
        all_objects.append({
            "name": str(data["name"]),
            "pc": data["point_cloud"],
            "id": i,
        })

    fig = plt.figure(figsize=(20, 16))

    views = [
        (221, "Top View (X-Y plane)", "X (m)", "Y (m)", 0, 1),
        (222, "Front View (X-Z plane)", "X (m)", "Z (depth, m)", 0, 2),
        (223, "Side View (Y-Z plane)", "Y (m)", "Z (depth, m)", 1, 2),
    ]

    for subplot_idx, title, xlabel, ylabel, dim_x, dim_y in views:
        ax = fig.add_subplot(subplot_idx)
        for obj, cm in zip(all_objects, COLORS_MPL):
            pc = obj["pc"]
            idx = np.random.choice(len(pc), min(2000, len(pc)), replace=False)
            ax.scatter(pc[idx, dim_x], pc[idx, dim_y], s=0.5, c=cm,
                       label=f'[{obj["id"]}] {obj["name"]}', alpha=0.7)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, markerscale=5)
        if dim_x == 0 and dim_y == 1:
            ax.axis("equal")

    # 3D
    ax3 = fig.add_subplot(224, projection='3d')
    for obj, cm in zip(all_objects, COLORS_MPL):
        pc = obj["pc"]
        idx = np.random.choice(len(pc), min(1500, len(pc)), replace=False)
        ax3.scatter(pc[idx, 0], pc[idx, 1], pc[idx, 2],
                    s=0.5, c=cm, label=f'[{obj["id"]}] {obj["name"]}', alpha=0.7)
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.set_title("3D Point Cloud", fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=7, markerscale=5)
    ax3.view_init(elev=25, azim=-60)

    plt.suptitle("Per-Object Point Clouds (MoGe Depth -> Camera Space)",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "step4_point_clouds.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [vis] 点云可视化 -> {out}")
    return out


# ===================================================================
# 全步骤可视化入口
# ===================================================================

def visualize_all_steps(input_dir: str, output_dir: str = None):
    """运行 test_02 全部四个步骤的可视化。

    Args:
        input_dir:  包含 test_02 输出数据的目录（object_*.npz, scene_data.npz 等）
        output_dir: 可视化输出目录，默认与 input_dir 相同
    """
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  [自动] 生成 test_02 各步骤可视化 ...")
    print("=" * 60)

    visualize_detections(input_dir, output_dir)
    visualize_segmentation(input_dir, output_dir)
    visualize_depth(input_dir, output_dir)
    visualize_point_clouds(input_dir, output_dir)

    print(f"\n  [vis] 全部可视化完成 -> {output_dir}/\n")
