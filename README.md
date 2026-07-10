# CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image

复现论文 [CAST (CVPR 2025)](https://arxiv.org/abs/2502.12894) 的代码实现。

> **Kaixin Yao\*, Longwen Zhang\*, Xinhao Yan, Yan Zeng, Qixuan Zhang†, Wei Yang, Lan Xu‡, Jiayuan Gu‡, Jingyi Yu‡**
> *CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image*

---

## ⚠️ 重要说明 — 实现状态

本文复现了 CAST 论文描述的**完整推理管线架构**，并使用 **Meta SAM 3D Objects** 替代了原论文的 ObjectGen + AlignGen，使用 **Qwen-VL** (DashScope API) 替代了原论文的 GPT-4V 进行场景关系图推理。

| 模块 | 状态 | 说明 |
|------|------|------|
| **SAM 3D Objects** | ✅ 主生成器 | `facebook/sam-3d-objects`，自动从 HuggingFace 下载（~7GB） |
| **PoseAdapter（坐标桥接+ICP精修）** | ✅ 完整实现 | 替代原 AlignGen，支持 none/umeyama/icp 三种精修模式 |
| **Qwen-VL 场景图推理** | ✅ 完整实现 | DashScope API，也兼容 GPT-4V（OpenAI 兼容格式） |
| **Florence-2 目标检测** | ✅ 完整实现 | 懒加载，自动从 HuggingFace 下载（~1.7GB） |
| **Grounding DINO 目标检测** | ✅ 完整实现 | 备选检测器 `grounding-dino-tiny`（~700MB），支持文本提示 |
| **GroundedSAM (SAM) 分割** | ✅ 完整实现 | 懒加载 `facebook/sam-vit-base`（~1.2GB），bbox-prompted mask |
| **MoGe 深度估计** | ✅ 完整实现 | 懒加载 `Ruicheng/moge-vitl`（~1.2GB），输出 metric depth + 内参 |
| **Depth-Anything-V2 深度估计** | ✅ 完整实现 | 备选深度估计器（~200MB），轻量化替代方案 |
| **Umeyama 变换求解** | ✅ 完整实现 | 闭式 SVD 解法 |
| **可微渲染对齐** | ✅ 完整实现 | PyTorch3D silhouette / 点云重投影 loss 双模式 |
| **物理感知校正** | ✅ 核心实现 | SDF 约束 + 6D 旋转表示 + PyTorch 优化 |
| **Set-of-Mark 可视化** | ✅ 完整实现 | 用于 VLM 视觉提示 |
| **关系图构建** | ✅ 完整实现 | 细粒度→粗粒度映射 + 多数投票集成 |

---

## 模型资源需求总览

### 🔑 需要 API Key（无需下载权重，调用云端 API）

| 模型 | 提供方 | API 地址 | 用途 | 费用 |
|------|--------|----------|------|------|
| **Qwen-VL** | 阿里云 DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 场景关系图推理（默认） | 按 token 计费 |
| **GPT-4V** | OpenAI | `https://api.openai.com/v1` | 场景关系图推理（备选） | 按 token 计费 |

### 📥 需要下载权重（本地 GPU 推理）

| 模型 | 来源 | 大小 | 用途 | 是否自动下载 |
|------|------|------|------|-------------|
| **SAM 3D Objects** | HuggingFace `facebook/sam-3d-objects` | ~7 GB | 物体网格+纹理+姿态生成 | ✅ 首次运行时自动下载 |
| **Florence-2** | HuggingFace `microsoft/Florence-2-large` | ~1.7 GB | 目标检测（默认） | ✅ 懒加载，首次调用时自动下载 |
| **Grounding DINO** | HuggingFace `IDEA-Research/grounding-dino-tiny` | ~700 MB | 目标检测（备选） | ✅ 懒加载，首次调用时自动下载 |
| **SAM** | HuggingFace `facebook/sam-vit-base` | ~1.2 GB | 语义分割 | ✅ 懒加载，首次调用时自动下载 |
| **MoGe** | HuggingFace `Ruicheng/moge-vitl` | ~1.2 GB | 深度估计（默认） | ✅ 懒加载，首次调用时自动下载 |
| **Depth-Anything-V2** | HuggingFace `depth-anything/Depth-Anything-V2-Small-hf` | ~200 MB | 深度估计（备选） | ✅ 懒加载，首次调用时自动下载 |

> **注意**: 所有模型均使用懒加载模式 — 仅在首次调用对应函数时才会下载和初始化。如果模型下载失败，会自动回退到 stub/placeholder，管线仍可运行但不保证精度。

### GPU 显存需求

| 组件 | 最低显存 | 推荐显存 |
|------|----------|----------|
| SAM 3D Objects (fp16) | ~12 GB | 16+ GB |
| SAM 3D Objects (fp32) | ~16 GB | 24+ GB |
| Florence-2 | ~3 GB | 4+ GB |
| SAM | ~3 GB | 4+ GB |
| MoGe | ~2.5 GB | 4+ GB |
| Grounding DINO (tiny) | ~1.5 GB | 2+ GB |
| Depth-Anything-V2 (small) | ~0.5 GB | 1+ GB |

> **注意**: 上述模型均为懒加载，仅在实际调用时占用显存。可通过设置环境变量 `CUDA_VISIBLE_DEVICES=""` 使用 CPU 推理（速度较慢）。

---

## 项目结构

```
SAMScene/
├── README.md                 # 本文件
├── requirements.txt          # Python 依赖
├── config.py                 # 配置类与默认参数（含 Qwen/GPT-4V 双 VLM 配置）
├── pipeline.py               # 主管线：串联所有阶段
├── scene_analysis.py         # Stage 1: 场景分析（检测/分割/深度/VLM）
├── sam3d_wrapper.py          # Stage 2a: SAM 3D 封装（替代原 ObjectGen）
├── pose_adapter.py           # Stage 2b: 坐标系桥接 + ICP 精修（替代原 AlignGen）
├── alignment.py              # Umeyama + ICP fallback（保留为底层工具）
├── iterative_procedure.py    # Stage 2c: SAM 3D 单步生成 + 精修
├── physics_correction.py     # Stage 4: 物理感知校正优化
├── utils/
│   ├── __init__.py
│   ├── sdf_utils.py          # SDF 计算与网格工具
│   ├── point_cloud.py        # FPS、归一化、去噪
│   └── relation_graph.py     # Set-of-Mark + VLM API 调用（Qwen/GPT-4V）
└── examples/
    └── demo.py               # 命令行 Demo
```

---

## 安装

### 1. 基础依赖

```bash
cd SAMScene
pip install -r requirements.txt
```

### 2. VLM API Key 配置

**方案 A: 使用 Qwen (推荐)**

```bash
# 注册阿里云 DashScope: https://dashscope.console.aliyun.com/
# 获取 API Key，然后:
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

**方案 B: 使用 GPT-4V**

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

### 3. 可选：本地模型权重

所有本地模型均使用**懒加载**模式 — 首次调用时自动从 HuggingFace 下载。如果网络问题导致下载失败，模型会自动回退到 stub（管线仍可运行）。

```bash
# Florence-2（目标检测 — 默认检测器）
# 模型: microsoft/Florence-2-large (~1.7 GB)
# 首次调用 detect_objects_florence2() 时自动下载

# SAM（语义分割 — bbox → mask 精修）
# 模型: facebook/sam-vit-base (~1.2 GB)
# 首次调用 segment_objects_grounded_sam() 时自动下载

# MoGe（深度估计 — 默认深度估计器）
# 模型: Ruicheng/moge-vitl (~1.2 GB)
# 安装: pip install moge
# 首次调用 estimate_depth_moge() 时自动下载

# 备选模型（可选）:
# - Grounding DINO: detect_objects_grounding_dino() 自动下载
# - Depth-Anything-V2: estimate_depth_metric3d() 自动下载
# - PyTorch3D 可微渲染: pip install pytorch3d
```

---

## 使用方法

### 最简运行（stub 模式，测试管线逻辑）

```bash
python examples/demo.py --image /path/to/image.jpg --quick --device cpu
```

### 完整运行（Qwen VLM + SAM 3D）

```bash
python examples/demo.py \
    --image /path/to/image.jpg \
    --vlm qwen \
    --qwen-key $DASHSCOPE_API_KEY \
    --qwen-model qwen-vl-max \
    --output ./output/my_scene
```

### 使用 GPT-4V 作为 VLM 后端

```bash
python examples/demo.py \
    --image /path/to/image.jpg \
    --vlm openai \
    --openai-key $OPENAI_API_KEY \
    --output ./output/my_scene
```

### Python API

```python
from config import CASTConfig
from pipeline import CASTPipeline

# 使用 Qwen VLM (默认)
config = CASTConfig(
    device="cuda",
    vlm_provider="qwen",
    qwen_api_key="sk-xxxxx",          # DashScope API key
    qwen_model="qwen-vl-max",         # 可选: qwen-vl-plus, qwen2.5-vl-72b-instruct
    max_iterations=1,
)

# 或使用 GPT-4V
config = CASTConfig(
    device="cuda",
    vlm_provider="openai",
    openai_api_key="sk-xxxxx",
    gpt_model="gpt-4-vision-preview",
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
  ├── VLM (Qwen/GPT-4V) → 过滤误检、保持开放词汇
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

| 方面 | 原 CAST | 当前实现 (SAMScene) |
|------|---------|-------------------|
| 网格生成 | ObjectGen (1.5B CLAY, 未开源) | SAM 3D (1.2B, 已开源) |
| 纹理 | 额外纹理模块 | 原生纹理输出 |
| 姿态估计 | AlignGen (150M, 点云扩散) | SAM 3D 6D layout (端到端) |
| 遮挡处理 | DINOv2 MAE 填充 | 架构内置 |
| 迭代次数 | 3 次 ObjectGen↔AlignGen | 1 次 + 可选 ICP 精修 |
| 点云条件 | 部分点云 cross-attn 注入 | 训练时隐式学习 |
| **场景图推理** | **GPT-4V (OpenAI)** | **Qwen-VL (DashScope) / GPT-4V 双后端** |

### Stage 3: 场景关系图（Section 5.3）

```
VLM (Qwen-VL / GPT-4V) + Set-of-Mark
  ├── 6 种细粒度关系类型:
  │   Stack | Lean | Hang | Clamped | Contained | Edge/Point
  ├── 集成策略（多数投票，≥半数即确认）
  └── 映射为粗粒度约束:
      Contact（双向） + Support（单向）
```

**VLM 配置说明：**
- **Qwen-VL**（默认）: 通过阿里云 DashScope 兼容模式，使用 OpenAI 兼容 API
  - 模型: `qwen-vl-max` / `qwen-vl-plus` / `qwen2.5-vl-72b-instruct`
  - Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **GPT-4V**（备选）: 使用 OpenAI 原生 API
  - 模型: `gpt-4-vision-preview` / `gpt-4o`

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

### VLM API 调用 (utils/relation_graph.py)

- 统一使用 OpenAI 兼容的 API 格式（`openai` Python 库）
- 支持 Qwen DashScope 和 GPT-4V 两种后端，切换仅需修改 `base_url` 和 `model`
- Set-of-Mark 视觉提示：随机彩色 mask + 数字编号
- 集成策略：3 次独立调用 + 多数投票（≥半数确认）
- 6 种细粒度关系 → Contact/Support 映射
- 鲁棒 JSON 解析：支持 markdown code fence、裸数组、混合文本

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

---

## 已知限制

1. **SAM 3D 依赖网络** — 首次运行时自动从 HuggingFace 下载约 7GB 模型。设置 `sam3d_offline=True` 或 `--sam3d-offline` 可跳过。

2. **深度估计精度** — 如果 MoGe 加载失败，会自动回退到恒值深度（2m）。建议安装 `pip install moge` 以获取真实深度。轻量备选方案 Depth-Anything-V2 仅提供相对深度，需启发式映射为度量值。

3. **VLM 依赖** — 关系图的质量直接依赖 Qwen/GPT-4V 的常识推理能力。无 API key 时返回空图，物理校正将跳过。

4. **SAM 分割精度** — 使用 bbox-prompted SAM 比全 GroundedSAMv2（text-prompted）精度略低。如需最佳结果，可替换为 Grounding DINO + SAM 的组合管线。

5. **纹理生成** — 论文 Section 4.3 末尾提到使用预训练的纹理模块（CLAY 的一部分）生成 UV 纹理。SAM 3D 自带纹理输出，但质量可能不同。

6. **训练代码** — 本仓库仅包含推理管线。ObjectGen 和 AlignGen 的训练遵循 3DShape2VecSet/CLAY 的方法。

7. **多视角一致性** — 在复杂场景（多物体重叠、密集排列）中，仅依赖单视图深度估计可能导致空间冲突。物理校正可在一定程度上缓解此问题。

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

本代码仅供学术研究使用。预训练模型权重（CLAY、MoGe、SAM 3D 等）遵循各自原始许可证。
