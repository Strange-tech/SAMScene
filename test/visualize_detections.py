#!/usr/bin/env python3
"""
可视化 Florence-2 目标检测结果：
  读取 test/output 中的检测数据，在图像上绘制检测框和文本标签，
  输出可视化图像到 output 目录。

用法:
  python visualize_detections.py --output ./test/output
"""

import argparse
import os
import sys

import cv2
import numpy as np

# 为不同类别分配不同颜色（BGR 格式）
COLORS = [
    (0, 255, 0),     # 绿色
    (255, 0, 0),     # 蓝色
    (0, 0, 255),     # 红色
    (255, 255, 0),   # 青色
    (255, 0, 255),   # 品红
    (0, 255, 255),   # 黄色
    (128, 0, 128),   # 紫色
    (255, 128, 0),   # 橙色
    (0, 128, 128),   # 深青
    (128, 128, 0),   # 橄榄
]


def visualize_detections(input_dir: str, output_dir: str):
    """读取检测结果并生成可视化图像"""

    # 1. 读取输入图像
    image_path = os.path.join(input_dir, "input_image.png")
    if not os.path.exists(image_path):
        print(f"[ERROR] 输入图像不存在: {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"  图像尺寸: {w} x {h}")

    # 2. 读取场景元数据
    import json
    meta_path = os.path.join(input_dir, "scene_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        num_objects = meta.get("num_objects", 0)
        print(f"  检测物体数: {num_objects}")
    else:
        # 自动扫描 object_*.npz 文件
        import glob
        npz_files = sorted(glob.glob(os.path.join(input_dir, "object_*.npz")))
        num_objects = len(npz_files)

    # 3. 读取每个物体的检测结果并绘制
    print(f"\n  Florence-2 检测结果:")
    print(f"  {'ID':<4} {'名称':<18} {'BBox (x1,y1,x2,y2)':<30} {'描述'}")
    print(f"  {'-'*4} {'-'*18} {'-'*30} {'-'*20}")

    for i in range(num_objects):
        npz_path = os.path.join(input_dir, f"object_{i:02d}.npz")
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path, allow_pickle=True)
        name = str(data["name"])
        description = str(data["description"])
        bbox = data["bbox"]  # [x1, y1, x2, y2]

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = COLORS[i % len(COLORS)]

        # 绘制边界框
        thickness = max(2, min(w, h) // 400)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签背景和文字
        label = f"[{i}] {name}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )

        # 标签放在 bbox 上方，如果超出图像则放在内部
        label_y = y1 - 5
        if label_y - text_h < 0:
            label_y = y1 + text_h + 5

        # 背景矩形
        cv2.rectangle(
            img,
            (x1, label_y - text_h - 4),
            (x1 + text_w + 6, label_y + 4),
            color,
            -1,  # 填充
        )

        # 白色文字
        cv2.putText(
            img, label,
            (x1 + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2,
            cv2.LINE_AA,
        )

        print(f"  {i:<4} {name:<18} [{x1:4d},{y1:4d},{x2:4d},{y2:4d}]    {description}")

    # 4. 在图像底部添加图例
    legend_y = h - 10
    legend_items_per_row = 4
    for idx in range(0, num_objects, legend_items_per_row):
        row_items = min(legend_items_per_row, num_objects - idx)
        start_x = 10
        for j in range(row_items):
            i = idx + j
            data = np.load(os.path.join(input_dir, f"object_{i:02d}.npz"), allow_pickle=True)
            name = str(data["name"])
            color = COLORS[i % len(COLORS)]

            legend_text = f"[{i}] {name}"
            (tw, th), _ = cv2.getTextSize(legend_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            # 色块
            cv2.rectangle(
                img,
                (start_x, legend_y - th),
                (start_x + 12, legend_y),
                color, -1,
            )
            # 文字
            cv2.putText(
                img, legend_text,
                (start_x + 16, legend_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            start_x += tw + 30

    # 5. 添加标题
    title = "Florence-2 Object Detection Results"
    cv2.putText(
        img, title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA,
    )

    # 6. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "florence2_detection_results.png")
    cv2.imwrite(output_path, img)
    print(f"\n  [OK] 可视化结果已保存: {output_path}")

    return output_path


def main():
    p = argparse.ArgumentParser(
        description="可视化 Florence-2 目标检测结果"
    )
    p.add_argument("--output", default="./test/output",
                   help="输出目录（同时读取此目录中的检测数据）")
    args = p.parse_args()

    print("=" * 60)
    print("  可视化 Florence-2 目标检测结果")
    print("=" * 60)

    input_dir = args.output  # 检测结果也在这个目录中
    visualize_detections(input_dir, args.output)


if __name__ == "__main__":
    main()
