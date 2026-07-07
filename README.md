# CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image

复现论文 [CAST (CVPR 2025)](https://arxiv.org/abs/2502.12894) 的代码实现。

> **Kaixin Yao\*, Longwen Zhang\*, Xinhao Yan, Yan Zeng, Qixuan Zhang†, Wei Yang, Lan Xu‡, Jiayuan Gu‡, Jingyi Yu‡**
> *CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image*

---

## ⚠️ 重要说明 — 实现状态

本文复现了 CAST 论文描述的**完整推理管线架构**，并使用 **Meta SAM 3D Objects** 替代了原论文的 ObjectGen + AlignGen。

| 模块 | 状态 | 说明 |
|------|------|------|
| **SAM 3D Objects** | ✅ 主生成器 | `facebook/sam-3d-objects`，自动从 HuggingFace 下载（~7GB） |
| **PoseAdapter（坐标桥接+ICP精修）** | ✅ 完整实现 | 替代原 AlignGen，支持 none/umeyama/icp 三种精修模式 |
| **Florence-2 目标检测** | ⚠️ stub | 需下载 `microsoft/Florence-2-large`（~1.7GB） |
| **GroundedSAMv2 分割** | ⚠️ stub | 需安装 `segment-anything` + 下载 SAM2 权重 |
| **MoGe 深度估计** | ⚠️ stub | 需安装 `moge` 并下载 `Ruicheng/moge-vitl` 权重 |
| **GPT-4V 关系推理** | ⚠️ stub | 需 OpenAI API key（GPT-4V 访问权限） |
| **Umeyama 变换求解** | ✅ 完整实现 | 闭式 SVD 解法 |
| **物理感知校正** | ✅ 核心实现 | SDF 约束 + 6D 旋转表示 + PyTorch 优化 |
| **Set-of-Mark 可视化** | ✅ 完整实现 | 用于 GPT-4V 视觉提示 |
| **关系图构建** | ✅ 完整实现 | 细粒度→粗粒度映射逻辑 |
| **FPS 采样 / 点云归一化** | ✅ 完整实现 | 2048 点 |

### 依赖外部资源总结

| 资源 | 用途 | 备注 |
|------|------|------|
| **SAM 3D Objects** | 物体生成+姿态 | 自动下载，需 ~16GB GPU |
| **Florence-2** | 目标检测 | 可选替代方案 |
| **SAM2** | 语义分割 | 可选替代方案 |
| **MoGe** | 深度估计 | 备用，SAM 3D 自带 pose |
| **GPT-4V API** | 关系图推理 | 无 key 时返回空图 |

---

## 项目结构

```
CAST/
├── README.md                 # 本文件
├── requirements.txt          # Python 依赖
├── config.py                 # 配置类与默认参数
├── pipeline.py               # 主管线：串联所有阶段
├── scene_analysis.py         # Stage 1: 场景分析（检测/分割/深度）
├── sam3d_wrapper.py          # Stage 2a: SAM 3D 封装（替代原 ObjectGen）
├── pose_adapter.py           # Stage 2b: 坐标系桥接 + ICP 精修（替代原 AlignGen）
├── alignment.py              # Umeyama + ICP fallback（保留为底层工具）
├── iterative_procedure.py    # Stage 2c: SAM 3D 单步生成 + 精修
├── physics_correction.py     # Stage 4: 物理感知校正优化
├── utils/
│   ├── __init__.py
│   ├── sdf_utils.py          # SDF 计算与网格工具
│   ├── point_cloud.py        # FPS、归一化、去噪
│   └── relation_graph.py     # Set-of-Mark + GPT-4V 提示词
└── examples/
    └── demo.py               # 命令行 Demo
```

---

## 安装

```bash
cd CAST
pip install -r requirements.txt
# 可选（用于本地推理的完整管线）:
# pip install transformers segment-anything groundingdino-py moge
```

---

## 使用方法

### 最简运行（stub 模式，测试管线逻辑）

```bash
python examples/demo.py --image /path/to/image.jpg --quick --device cpu
```

### 使用预训练权重

```bash
python examples/demo.py \
    --image /path/to/image.jpg \
    --objectgen-ckpt /path/to/clay_checkpoint.pt \
    --aligngen-ckpt /path/to/aligngen_checkpoint.pt \
    --openai-key sk-xxxxx \
    --output ./output/my_scene
```

### Python API

```python
from config import CASTConfig
from pipeline import CASTPipeline

config = CASTConfig(
    device="cuda",
    max_iterations=3,
    objectgen_ckpt="/path/to/clay.pt",     # None = stub
    aligngen_ckpt="/path/to/aligngen.pt",  # None = ICP fallback
    openai_api_key="sk-xxxxx",             # None = skip GPT-4V
)

pipe = CASTPipeline(config)
scene = pipe.reconstruct("my_image.jpg", output_dir="./output")
scene.export("./output")
```

---

## 论文方法对照

### Stage 1: 场景分析（Section 3）

```
输入图像
  ├── Florence-2 → 目标检测 + 描述 + 边界框
  ├── GPT-4v → 过滤误检、保持开放词汇
  ├── GroundedSAMv2 → 精细化分割掩膜 {M_i}
  ├── MoGe → 像素对齐深度图 + 相机内参
  └── 深度→点云 → 逐物体点云 {q_i}（场景坐标系）
```

### Stage 2: 感知 3D 实例生成（Section 4 — 已用 SAM 3D 替换）

```
SAM 3D 单步推理 (替代原 ObjectGen + AlignGen):
  Step 1 - SAM 3D:
    mesh + texture + (R, t, s)_cam = SAM3D(image ⊙ occlusion_mask)
    内部: DINOv2 → 1.2B Flow Matching Transformer (MoT)
          → voxel shape + 6D layout → sparse latent refinement
          → VAE decode → textured mesh

  Step 2 - PoseAdapter (替代原 AlignGen):
    (R, t, s)_scene = PoseAdapter.adapt(mesh, R_cam, t_cam, s_cam, scene_pc)
    桥接: 相机坐标系 → 场景坐标系
    可选: ICP/Umeyama 精修对齐场景点云
```

#### 与原 CAST 的对比

| 方面 | 原 CAST | 当前实现 (SAM 3D) |
|------|---------|-------------------|
| 网格生成 | ObjectGen (1.5B CLAY, 未开源) | SAM 3D (1.2B, 已开源) |
| 纹理 | 额外纹理模块 | 原生纹理输出 |
| 姿态估计 | AlignGen (150M, 点云扩散) | SAM 3D 6D layout (端到端) |
| 遮挡处理 | DINOv2 MAE 填充 | 架构内置 |
| 迭代次数 | 3 次 ObjectGen↔AlignGen | 1 次 + 可选 ICP 精修 |
| 点云条件 | 部分点云 cross-attn 注入 | 训练时隐式学习 |

### Stage 3: 场景关系图（Section 5.3）

```
GPT-4V + Set-of-Mark
  ├── 6 种细粒度关系类型:
  │   Stack | Lean | Hang | Clamped | Contained | Edge/Point
  ├── 集成策略（多数投票，≥半数即确认）
  └── 映射为粗粒度约束:
      Contact（双向） + Support（单向）
```

### Stage 4: 物理感知校正（Section 5.2–5.4）

```
SDF 约束图 + PyTorch 优化
  ├── Contact Cost (Eq. 9):
  │   C(i→j) = 穿透惩罚 + 最小距离惩罚
  │   双向: C(T_i, T_j) = C(i→j) + C(j→i)
  │
  ├── Support Cost (Eq. 10):
  │   单向: 仅优化被支撑物体
  │   近表面正则化 (Eq. 11): σ 带内鼓励紧密接触
  │
  └── 优化变量:
      6D 连续旋转表示 (Zhou et al. 2019)
      + 3D 平移向量
```

---

## 关键实现细节

### Umeyama 算法（alignment.py）
- 闭式 SVD 求解相似变换（旋转 R, 平移 t, 均匀缩放 s）
- 支持反射检测与修正
- 论文 Section 4.2 描述的标准方法

### 6D 旋转表示（physics_correction.py）
- 使用 Zhou et al. 2019 的连续 6D 表示
- 避免欧拉角/四元数的万向节锁和不连续性
- Gram-Schmidt 正交化 → 3×3 旋转矩阵

### FPS 采样（utils/point_cloud.py）
- 2048 点（与论文一致的默认值）
- 用于 ObjectGen 点云条件输入
- 用于物理校正的网格表面采样

### 点云扰动增强（object_generation.py）
- 训练时：α·GT + (1-α)·估算深度（α ∈ [0,1]）
- 随机掩膜（圆形/矩形）模拟遮挡
- 在运行时通过 `beta` 参数控制条件强度

---

## 已知限制

1. **Stub 生成器产生的形状不真实** — ObjectGen stub 是简单的 MLP，输出粗网格。需 CLAY 权重才能获得论文级别的几何质量。

2. **深度估计精度** — MoGe stub 返回恒值深度。真实 MoGe 对于无约束图像能提供合理度量深度。

3. **GPT-4V 依赖** — 关系图的质量直接依赖 VLM 的常识推理。本实现提供了完整的提示词和解析逻辑，但无 API key 则返回空图。

4. **纹理生成** — 论文 Section 4.3 末尾提到使用预训练的纹理模块（CLAY 的一部分）生成 UV 纹理。本实现暂不包含。

5. **训练代码** — 本仓库仅包含推理管线。ObjectGen 和 AlignGen 的训练遵循 3DShape2VecSet/CLAY 的方法，需要 Objaverse 数据集（~500K 资产）和 GPU 集群。

6. **多视角一致性** — 在复杂场景（多物体重叠、密集排列）中，基于 stub 的管线性能会下降。

---

## 引用

如果本代码对你的研究有帮助，请引用原论文：

```bibtex
@article{yao2025cast,
  title={CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image},
  author={Yao, Kaixin and Zhang, Longwen and Yan, Xinhao and Zeng, Yan and
          Zhang, Qixuan and Yang, Wei and Xu, Lan and Gu, Jiayuan and Yu, Jingyi},
  journal={CVPR},
  year={2025}
}
```

---

## 许可

本代码仅供学术研究使用。预训练模型权重（CLAY、MoGe 等）遵循各自原始许可证。
