# Prism-Multimodal-Medical-Image-Reconstruction-via-Foundation-Model-Fine-Tuning
FuseRecon 是一套面向医学影像的通用重建框架，通过对大规模基础模型（Foundation Model）进行参数高效微调（如 LoRA / Adapter），实现跨模态（MRI、CT、PET 等）的图像重建与补全，在加速采样、低剂量、跨域迁移等场景下显著提升重建质量与泛化能力。
# FuseRecon

> **Multimodal Medical Image Reconstruction via Foundation Model Fine-Tuning**
>
> 基于大模型微调的医学图像多模态重建

[![Status](https://img.shields.io/badge/status-paper%20in%20preparation-orange)](https://github.com/)
[![License](https://img.shields.io/badge/license-TBD-blue)](./LICENSE)

> ⚠️ **本 README 为项目早期草稿**，对应论文正在撰写中，部分章节（实验结果、复现命令、架构图）为占位内容，待论文定稿后补全真实数字与图表。

---

## 📌 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [支持模态](#支持模态)
- [方法概述](#方法概述)
- [环境依赖](#环境依赖)
- [安装](#安装)
- [快速开始](#快速开始)
- [微调（Fine-Tuning）](#微调fine-tuning)
- [评测](#评测)
- [实验结果](#实验结果占位)
- [项目结构](#项目结构)
- [路线图](#路线图)
- [引用](#引用)
- [许可证](#许可证)
- [联系方式](#联系方式)

---

## 项目简介

FuseRecon 是一套面向医学影像的**通用重建框架**。其核心思路是：以预训练的基础模型（Foundation Model）为骨干，通过**参数高效微调**（Parameter-Efficient Fine-Tuning, PEFT，如 LoRA / Adapter）将其适配到多模态重建任务，从而在加速采样、低剂量、跨域迁移等 challenging 场景下，显著提升重建质量与跨模态泛化能力。

传统医学图像重建通常依赖**单模态、手工设计先验**，难以在不同成像设备/模态间迁移。FuseRecon 通过在冻结骨干上注入可训练适配器，让 MRI、CT、PET 等模态**共享同一套学习到的表征**，既降低了单模态训练成本，又获得了更强的 few-shot / zero-shot 跨模态迁移能力。

---

## 核心特性

- 🔬 **多模态统一**：一套框架覆盖 MRI / CT / PET 等模态的重建与补全。
- 🧠 **基础模型驱动**：复用大规模预训练表征，告别从零训练。
- ⚡ **参数高效微调**：LoRA / Adapter 注入，训练成本低、可多任务并行。
- 🔁 **跨模态迁移**：支持 unseen 模态的 zero-/few-shot 重建。
- 📊 **标准化评测**：内置 PSNR / SSIM / NMSE 等指标与可视化管线。

---

## 支持模态

| 模态 | 任务示例 | 状态 |
|------|----------|------|
| MRI | 加速成像（欠采样 k-space 重建）、超分辨率 | ✅ |
| CT  | 低剂量重建、稀疏视角重建 | ✅ / 🚧 |
| PET | 低计数重建、去噪 | 🚧 |
| X-ray | 稀疏视角 / limited-angle | 📋 规划中 |

> ✅ 已支持　🚧 开发中　📋 规划中

---

## 方法概述

> 下方为文字版流程，正式架构图将在论文定稿后补入 `docs/figures/architecture.png`。

```
                 ┌─────────────────────────────┐
   多模态输入 ──▶ │   Frozen Foundation Backbone │
  (MRI/CT/PET)   │   (预训练基础模型, 参数冻结)   │
                 └──────────────┬──────────────┘
                                │
                  注入可训练 Adapter / LoRA
                                │
                 ┌──────────────▼──────────────┐
                 │   Reconstruction Head        │
                 │  (模态特定输出头, 可学习)      │
                 └──────────────┬──────────────┘
                                │
                  高质量重建图像 + 跨模态表征
```

**训练范式**：
1. 冻结基础模型骨干，仅训练注入的适配器参数；
2. 多模态数据混合训练，共享表征、解耦输出头；
3. 对 unseen 模态执行 few-shot 微调或零参数推理。

---

## 环境依赖

- Python ≥ 3.9
- PyTorch ≥ 2.0
- 深度学习框架：_<占位：PyTorch Lightning / MONAI / 自研>_
- 微调工具：_<占位：PEFT / 自研 LoRA>_
- GPU：建议 ≥ 24GB 显存（单卡 RTX 3090 / A10 可跑实验配置）

> 完整 `requirements.txt` 见仓库根目录（论文定稿后随代码开源发布）。

---

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/<your-org>/FuseRecon.git
cd FuseRecon

# 2. 创建虚拟环境
python -m venv .venv && source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 快速开始

```python
# 推理示例（伪代码，待补真实 API）
from fuserecon import FuseReconModel

model = FuseReconModel.from_pretrained("fuserecon-base")
recon = model.reconstruct(undersampled_input, modality="MRI")
```

```bash
# 命令行推理（待补真实入口）
python inference.py \
    --input ./data/raw/mri_undersampled.nii.gz \
    --modality MRI \
    --ckpt ./checkpoints/fuserecon-base.pt \
    --output ./outputs/recon.nii.gz
```

---

## 微调（Fine-Tuning）

```bash
# LoRA 微调示例（占位命令，参数待定）
python train.py \
    --backbone <foundation-model-name> \
    --method lora \
    --lora-rank 8 \
    --modalities MRI CT \
    --data-root ./data/processed \
    --batch-size 8 \
    --epochs 50 \
    --output-dir ./checkpoints/fuserecon-lora
```

**关键超参（待论文补全真实取值）**

| 参数 | 说明 | 默认/取值 |
|------|------|-----------|
| `backbone` | 基础模型 | _占位_ |
| `method` | 微调策略 | lora / adapter |
| `lora_rank` | LoRA 秩 | _占位_ |
| `modalities` | 参与训练的模态 | MRI, CT, ... |

---

## 评测

```bash
python eval.py \
    --ckpt ./checkpoints/fuserecon-lora/last.pt \
    --test-root ./data/test \
    --metrics psnr ssim nmse \
    --save-vis ./outputs/vis
```

---

## 实验结果（占位）

> 以下为模板，论文定稿后替换为真实数字。

| 方法 | 模态 | 加速比 | PSNR ↑ | SSIM ↑ | NMSE ↓ |
|------|------|--------|--------|--------|--------|
| Baseline (单模态) | MRI R=4 | 4× | _xx.xx_ | _0.xxx_ | _x.xxx_ |
| FuseRecon (ours) | MRI R=4 | 4× | **_xx.xx_** | **_0.xxx_** | **_x.xxx_** |
| FuseRecon (few-shot) | CT | — | _xx.xx_ | _0.xxx_ | _x.xxx_ |

**定性结果**：见 `docs/figures/results.png`（待补）。

---

## 项目结构

```
FuseRecon/
├── fuserecon/            # 核心代码
│   ├── models/           # 骨干 + Adapter + 重建头
│   ├── data/             # 多模态数据加载与预处理
│   ├── train.py          # 微调入口
│   ├── eval.py           # 评测入口
│   └── inference.py      # 推理入口
├── configs/              # 训练/评测配置
├── data/                 # 数据集（软链接或说明）
├── checkpoints/          # 权重（开源后发布）
├── docs/                 # 文档与图表
├── requirements.txt
└── README.md
```

---

## 路线图

- [x] 多模态统一训练框架
- [x] MRI 加速重建基线
- [ ] CT / PET 模态扩展
- [ ] 跨模态 zero-shot 迁移评测
- [ ] 论文投稿（MICCAI / MIDL / IEEE TMI）
- [ ] 代码与权重开源

---

## 引用

> 论文撰写中，以下为占位引用格式，定稿后更新。

```bibtex
@article{fuserecon2026,
  title     = {FuseRecon: Multimodal Medical Image Reconstruction via Foundation Model Fine-Tuning},
  author    = {<Your Name> and <Co-authors>},
  journal   = {To appear},
  year      = {2026}
}
```

---

## 许可证

TBD（计划采用 _<Apache-2.0 / MIT>_，以开源发布时为准）。

---

## 联系方式

- 项目维护：_<你的名字 / 邮箱>_
- 问题反馈：请在 GitHub Issues 提交。

---

<p align="center">
  <sub>FuseRecon · 让多模态医学重建，交给微调后的大模型。</sub>
</p>
