# SAMScene: 3D Scene Reconstruction from a Single Image

基于 [CAST (CVPR 2025)](https://arxiv.org/abs/2502.12894) 的复现实现，使用 [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) 作为 3D 生成后端，支持 [MIDI-3D](https://github.com/VAST-AI/MIDI-3D) 作为备选模型。

---

## 测试进度

| 测试 | 阶段 | 状态 |
|------|------|------|
| test_01 | 图像加载与验证 | ✅ |
| test_02 | 场景分析（检测 + 分割 + 深度估计） | ✅ |
| test_03 | 物体生成（SAM 3D / MIDI） | ✅ |
| test_04 | 场景关系图（VLM） | ⏳ |
| test_05 | 物理感知校正 | ⏳ |
| test_06 | 场景导出 | ⏳ |

---

## 环境配置

### 1. 基础环境

```bash
conda create -n samscene python=3.10
conda activate samscene
pip install -r requirements.txt
```

### 2. SAM 3D Objects

```bash
# 克隆官方仓库
git clone https://github.com/facebookresearch/sam-3d-objects.git ../sam3d_repo
cd ../sam3d_repo
pip install -r requirements.txt
pip install -r requirements.inference.txt

# 下载权重（需 HuggingFace token 授权 gated repo）
huggingface-cli download facebook/sam-3d-objects \
    --local-dir ../SAMScene/pretrained_weights/sam3d \
    --token hf_xxx
```

权重目录结构：
```
pretrained_weights/sam3d/checkpoints/
    pipeline.yaml
    ss_generator.ckpt / ss_decoder.ckpt / ss_encoder.ckpt / ss_encoder.safetensors
    slat_generator.ckpt / slat_decoder_gs.ckpt / slat_decoder_gs_4.ckpt
    slat_decoder_mesh.ckpt / slat_decoder_mesh.pt
    slat_encoder.ckpt
    各模块对应的 .yaml 配置文件
```

### 3. MIDI-3D（可选）

```bash
git clone https://github.com/VAST-AI/MIDI-3D.git ../MIDI-3D
# 权重放于 pretrained_weights/midi_3d/
```

### 4. VLM API（test_04 需要）

```bash
export DASHSCOPE_API_KEY="sk-xxx"   # Qwen-VL（默认）
export OPENAI_API_KEY="sk-xxx"      # GPT-4V（备选）
```

---

## 测试命令

```bash
# 激活环境
conda activate samscene

# test_02: 场景分析（Florence-2 目标检测 + SAM 分割 + MoGe 深度估计）
python test/test_02_scene_analysis.py \
    --image data/images/room.png \
    --output ./test/output

# test_03: 物体生成
# 使用 SAM 3D（默认，需 GPU 16GB+）
python test/test_03_object_generation.py \
    --model sam3d \
    --output ./test/output

# 使用 MIDI（备选）
python test/test_03_object_generation.py \
    --model midi \
    --output ./test/output

# test_04-06: 后续阶段（待完成）
python test/test_04_relation_graph.py --output ./test/output
python test/test_05_physics_correction.py --output ./test/output
python test/test_06_export.py --output ./test/output
```

---

## 参考文献

1. **CAST** — Kaixin Yao, Longwen Zhang, et al. *CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image.* CVPR 2025.
2. **SAM 3D Objects** — Meta Superintelligence Labs. *SAM 3D Objects: 3DFY Anything in Images.* 2025.
3. **MIDI** — VAST-AI. *MIDI-3D: Multi-Instance Diffusion for 3D Scene Generation.* CVPR 2025.
