# RiverSeg — 水体分割系统架构文档

> 基于 **GroundingDINO + LoRA-SAM** 的开放词汇语义分割系统，专用于遥感水体提取。

---

## 1. 系统概述

### 1.1 整体架构

本系统采用**两阶段级联架构**（检测 → 分割），结合开放词汇目标检测与高效微调分割模型：

```
┌─────────────────────────────────────────────────────────────────────┐
│                          推理 / 测试流程                              │
│                                                                     │
│  文本提示 (e.g. "waterbody")                                        │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────┐      ┌──────────────────────────────┐        │
│  │  Grounding DINO  │─────▶│  框坐标 → SAM 坐标系变换     │        │
│  │  (开放词汇检测)    │      │  (原始坐标→1024×1024)        │        │
│  └──────────────────┘      └──────────┬───────────────────┘        │
│                                       │                             │
│                                       ▼                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    LoRA-SAM 语义分割                           │  │
│  │  ┌──────────┐   ┌────────────────┐   ┌────────────────────┐  │  │
│  │  │图像编码器  │──▶│ 提示编码器      │──▶│ 掩码解码器          │  │  │
│  │  │(ViT-B +   │   │ (从框生成嵌入)  │   │ (生成分割掩码)      │  │  │
│  │  │ LoRA q/v) │   └────────────────┘   └────────────────────┘  │  │
│  │  └──────────┘                                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                       │                             │
│                                       ▼                             │
│                                  分割掩码                            │
│                            (Softmax → ArgMax)                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 两种运行模式

| 模式 | 描述 | 适用场景 |
|------|------|---------|
| **离线预计算模式** | 先使用 `precompute_gd_boxes.py` 对全部数据集运行 GroundingDINO，将检测结果保存为 `.txt` 标签文件；训练/推理时直接从文件读取 | 训练时需要固定标注 |
| **在线推理模式** | 推理时动态调用 GroundingDINO，从文本提示实时生成检测框 | 对任意新图像灵活推理 |

---

## 2. 第一阶段：GroundingDINO（开放词汇目标检测）

### 2.1 模型架构

GroundingDINO 基于 **DINO**（DETR with Improved Denoising Anchor Boxes）架构，融合了文本编码能力，实现开放词汇目标检测。

#### 2.1.1 整体结构

```
输入图像 ──▶ Swin Transformer ──▶ 多尺度特征 ──▶ 1×1 Conv 投影 ──▶ Transformer 编码器
                                                                │
输入文本 ──▶ BERT ──▶ 特征映射 ──▶──────────────────────────────────┤
                                                                │
                                                                ▼
                                                    Transformer 解码器
                                                    (含文本交叉注意力)
                                                                │
                                                                ▼
                                                    预测头: 分类 + 回归
                                                    ContrastiveEmbed + MLP
```

**位置**: `E:\work\seg\GroundingDINO\groundingdino\models\GroundingDINO\groundingdino.py`

#### 2.1.2 配置文件（Swin-Base）

```python
# 文件: groundingdino/config/GroundingDINO_SwinB_cfg.py
backbone = "swin_B_384_22k"       # 骨干网络 (Swin-B, 预训练尺寸384)
position_embedding = "sine"       # 正弦位置编码
hidden_dim = 256                   # Transformer 隐层维度
enc_layers = 6                     # 编码器层数
dec_layers = 6                     # 解码器层数
nheads = 8                         # 注意力头数
num_queries = 900                  # 检测查询数
num_feature_levels = 4             # 多尺度特征层数
two_stage_type = "standard"        # 两阶段(编码器输出选择候选)
text_encoder_type = "bert-base-uncased"  # 文本编码器
```

#### 2.1.3 核心组件

##### 骨干网络（Backbone）

- **Swin Transformer**（默认 `swin_B_384_22k`）或 **ResNet50/101**
- 输出多尺度特征图（`return_interm_indices = [1, 2, 3]`），对应 3 个层级
- 每个尺度的特征经过 `Conv2d(1×1) + GroupNorm` 投影到统一维度（`hidden_dim = 256`）

**文件**: 
- `.../backbone/backbone.py` — `Backbone` / `Joiner` 类
- `.../backbone/swin_transformer.py` — Swin Transformer 实现
- `.../backbone/position_encoding.py` — `PositionEmbeddingSineHW` 位置编码

##### 文本编码器（Text Encoder）

- 基于 **BERT-base-uncased**（`BertModelWarper` 包装）
- 处理流程：
  1. Tokenizer 将文本转换为 token IDs
  2. `generate_masks_with_special_tokens_and_transfer_map()` 生成：
     - `text_self_attention_masks` — 文本自注意力掩码（按 "." 分隔的片段内可见）
     - `position_ids` — 位置编码（每个片段内独立编号）
     - `cate_to_token_mask_list` — 类别到 token 的映射
  3. 输出经 `nn.Linear(768 → 256)` 投影到 Transformer 隐层维度

**文件**: `.../bertwarper.py`

##### 交叉模态融合（BiAttentionBlock）

在 Transformer 编码器中，使用 `BiAttentionBlock` 实现图像-文本的双向注意力：

```
图像特征 ↔ 文本特征     (双向交叉注意力)
```

- 图像特征与文本特征分别经过 LayerNorm
- `BiMultiHeadAttention` 计算两种模态间的注意力：
  - 图像到文本的注意力更新图像特征
  - 文本到图像的注意力更新文本特征
- 使用 DropPath + LayerScale 增强训练稳定性

**文件**: `.../fuse_modules.py` — `BiAttentionBlock` / `BiMultiHeadAttention`

##### Transformer 编码器（Encoder）

- 6 层 `DeformableTransformerEncoderLayer`
- 核心：**MultiScaleDeformableAttention**（多尺度可变形注意力）
  - 对每个查询点，在多个特征层上采样少量偏移点
  - 显著降低计算量，同时捕获多尺度信息
- 每层包含：可变形自注意力 → FFN (Linear-GELU-Linear)

**文件**: `.../transformer.py` — `TransformerEncoder` / `DeformableTransformerEncoderLayer`

##### Transformer 解码器（Decoder）

- 6 层 `DeformableTransformerDecoderLayer`
- 解码器结构（每层）：
  1. **自注意力**（查询间）
  2. **文本交叉注意力**（查询 ↔ 文本特征）
  3. **可变形交叉注意力**（查询 ↔ 图像多尺度特征）
  4. **FFN**
- 锚框迭代优化：每层基于上一层的输出预测 delta 偏移量

##### 预测头

- **分类头**: `ContrastiveEmbed` — 查询嵌入与文本嵌入的点积，得到类别 logits
- **回归头**: 3 层 MLP → 4 维 (cx, cy, w, h)

**文件**: `.../utils.py` — `ContrastiveEmbed` / `MLP`

### 2.2 推理 API

```python
# 核心推理函数 (groundingdino/util/inference.py)
def predict(model, image, caption, box_threshold, text_threshold):
    outputs = model(image, captions=[caption])   # 前向推理
    logits = outputs["pred_logits"].sigmoid()   # 分类置信度 (nq, 256)
    boxes = outputs["pred_boxes"]               # 归一化框坐标 (nq, 4)
    # 阈值过滤，获取有效框
```

---

## 3. 第二阶段：LoRA-SAM（语义分割）

### 3.1 模型架构

基于 **SAM（Segment Anything Model）** 的分割模型，使用 **LoRA（Low-Rank Adaptation）** 高效微调图像编码器。

#### 3.1.1 整体结构

```
输入图像 (1024×1024)
         │
         ▼
┌──────────────────┐     冻结参数
│ 图像编码器        │═══════════════════▶
│ (ViT-B, 12层)    │ ═══▶ qkv 投影 + LoRA (q/v)  ★可训练★
│  PatchEmbed       │
│  Block ×12        │
│  窗口/全局注意力   │
│  相对位置编码     │
└────────┬─────────┘
         │ 图像嵌入 (256×64×64)
         ▼
┌──────────────────┐  ★完全可训练★
│ 提示编码器        │◀──── 边界框提示 (B×N×4)
│  PositionEmbed    │
│  RandomFourier    │
│  × 2+MLP         │
└────────┬─────────┘
         │ 稀疏嵌入 (N×256)
         ▼
┌──────────────────┐  ★完全可训练★
│ 掩码解码器        │
│  TwoWayTransformer│
│  (2层, 8头)      │
│  上采样: ConvT×2  │
│  MLP → 掩码      │
└────────┬─────────┘
         │
         ▼
     分割输出 (B×C×256×256)
     (Softmax → ArgMax)
```

**文件**: `sam_lora_image_encoder.py` — `LoRA_Sam` / `_LoRA_qkv`

#### 3.1.2 LoRA 低秩适配

LoRA 仅对图像编码器中 **所有 12 个 Transformer 块的 q 和 v 投影** 注入低秩适配器（k 投影冻结）。

```
原始: qkv = W_qkv × x          [W_qkv: 768×2304]
LoRA: qkv' = qkv + Δq + Δv
       Δq = B_q × A_q × x     [A_q: r×768, B_q: 768×r]
       Δv = B_v × A_v × x     [A_v: r×768, B_v: 768×r]
```

参数配置（默认 `r = 4`）：

| 组件 | 原始参数量 | 可训练参数量 | 占比 |
|------|-----------|-------------|------|
| 图像编码器 | 91.3M | ~0.5M (LoRA) | 0.55% |
| 提示编码器 | 1.2M | 1.2M (全量) | 100% |
| 掩码解码器 | 4.1M | 4.1M (全量) | 100% |
| **总计** | ~97M | ~**5.8M** | ~**6%** |

权重文件约 **15MB**（vs 全量微调 ~374MB）

初始化策略（标准 LoRA 方案）：
- A 矩阵：Kaiming Uniform 初始化
- B 矩阵：零初始化（保证初始输出与原始模型一致）

#### 3.1.3 核心组件

##### 图像编码器（Image Encoder）

**位置**: `MobileSAM/mobile_sam/modeling/image_encoder.py`

- **PatchEmbed**: 16×16 Conv2d → 768 维嵌入
- **12 个 Transformer Block**（交替窗口注意力 / 全局注意力）：
  - 全局注意力层索引: `[2, 5, 8, 11]`（第 3, 6, 9, 12 层）
  - 窗口注意力层: 其余 8 层, 窗口大小 14
  - 相对位置编码（窗口注意力内）
- **Neck**: Conv2d(1×1) → LayerNorm2d → Conv2d(3×3) → LayerNorm2d
  - 输出通道: 256（适配提示编码器维度）

##### 提示编码器（Prompt Encoder）

**位置**: `MobileSAM/mobile_sam/modeling/prompt_encoder.py`

- 输入：边界框坐标 (N×4)，归一化到 [0, 1024]
- 编码方式：位置编码（Random Fourier Features）×2 + MLP → 256 维嵌入
- 输出：稀疏嵌入，与图像特征在掩码解码器中交互

##### 掩码解码器（Mask Decoder）

**位置**: `MobileSAM/mobile_sam/modeling/mask_decoder.py`

- **TwoWayTransformer**: 2 层, 8 头, 256 隐层维度
  - 交替自注意力 + 交叉注意力（token↔图像特征）
- 上采样：ConvTranspose2d 将低分辨率 (256×256) → 原图尺寸
- 输出：`C` 个掩码通道（`C = num_classes + 1`，含背景）

**关键修改**：对比原始 SAM 的 3 个 multimask 输出，本系统将 `num_multimask_outputs` 设为 `num_classes`，返回所有类别的掩码（不做切片过滤）。

#### 3.1.4 前向流程

```python
def forward(self, image, prompt, multimask_output, image_size):
    # 1. 预处理: 归一化(pixel_mean/std), 填充为正方形
    # 2. 图像编码器 → 图像嵌入 (B, 256, 64, 64)
    # 3. 提示编码器 → 稀疏嵌入 (B, N, 256)
    # 4. 掩码解码器 → 低分辨率掩码 (B, C, 256, 256)
    # 5. 后处理: 上采样到输入尺寸
    return {"masks": output_masks}  # (B, C, H, W)
```

### 3.2 损失函数

组合损失：`Loss = (1 - α) × FocalLoss + α × DiceLoss`，其中 `α = 0.75`（默认）

#### Focal Loss

```python
# utils.py: Focal_loss
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
# α = 0.25（背景类）, 1-α = 0.75（前景类）
# γ = 2
```

- 缓解类别不平衡（水体像素远少于背景）
- 聚焦于难分样本

#### Dice Loss

```python
# utils.py: DiceLoss
Dice(P, G) = 1 - (2 × |P∩G| + smooth) / (|P|² + |G|² + smooth)
```

- 衡量预测与真值的空间重叠程度
- 对像素级不均衡不敏感

---

## 4. 数据流

### 4.1 数据集结构

```
data/VOCdevkit/VOC2012/
├── JPEGImages/           # RGB 图像 (*.jpg)
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── SegmentationClass/    # 真值语义分割掩码 (*.png)
│   ├── 00001.png
│   └── ...
├── ImageSets/
│   └── Segmentation/     # 数据集划分文件
│       ├── train.txt     # 训练集 (683 images)
│       ├── val.txt       # 验证集 (195 images)
│       └── test.txt      # 测试集 (98 images)
├── TxtLabel/             # 手动标注的边界框 (*.txt)
│   └── ...
└── GD_Label/             # GroundingDINO 预生成边界框 (*.txt)
    └── ...
```

**文件**: `datasets/dataset_custom.py` — `MultiClassVOCSegmentation`

### 4.2 数据处理流程

```
训练/测试样本
    │
    ▼
┌─────────────────────┐
│ 加载图像 (PIL)       │
│ 加载掩码 (PIL)       │
│ 加载框文件 (.txt)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 框预处理              │
│ - 解析 N×4 数值      │
│ - 多框取面积最大框    │
│ - 保留 1×4 张量      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 数据增强 (训练)       │
│ - Resize (1024×1024)│
│ - RandomHFlip (0.5) │
│ - ToTensor          │
│ - Normalize         │
│   (mean/std =       │
│    ImageNet 参数)   │
└─────────┬───────────┘
          │
          ▼
    (image, mask, box)
```

对于训练集，图像统一缩放到 `1024×1024`，真值掩码保持对应尺寸。框坐标在 `1024×1024` 坐标系下。

### 4.3 GroundingDINO 离线预计算

```bash
# precompute_gd_boxes.py 的工作流程
1. 加载 GroundingDINO 模型 (Swin-B)
2. 遍历 train/val/test 中所有图像
3. 对每张图像调用 generate_boxes("waterbody")
4. 将检测结果保存为 TxtLabel 格式的 .txt 文件 (每行: x1,y1,x2,y2)
5. 输出到 GD_Label/ 目录
```

这样训练时可以直接读取预生成的框，无需在训练循环中运行 GroundingDINO，大幅提升训练速度。

**文件**: `precompute_gd_boxes.py`
**配置**: `--box_threshold 0.25 --text_threshold 0.25`

---

## 5. 训练系统

### 5.1 训练入口

```bash
python train.py \
    --batch_size 1 \
    --max_epochs 10 \
    --img_size 1024 \
    --num_classes 1 \
    --rank 4 \
    --base_lr 0.001 \
    --AdamW \
    --dice_param 0.75 \
    --lr_exp 0.9 \
    --label_dir data/VOCdevkit/VOC2012/GD_Label
```

**文件**: `train.py`

### 5.2 训练循环

```
训练循环 ├── DataLoader (batch_size=4, shuffle, 683 train imgs)
         │   每 epoch: 171 iterations
         │
         ├── 前向: LoRA-SAM(image, prompt) → masks
         │
         ├── 损失: (1-α)*FocalLoss + α*DiceLoss
         │
         ├── 反向传播 & 优化 (AdamW, lr=0.001)
         │
         ├── LR 调度:
         │   ├── Warmup (可选, 250 iter): 线性增长 0→base_lr
         │   └── 多项式衰减: lr = base_lr × (1 - t/T)^lr_exp
         │
         └── 每 epoch 结束后验证:
              ├── 遍历 val set (195 imgs)
              ├── 计算 mIoU, mAcc, oAcc
              ├── 保存 best model (mIoU 提升时)
              └── Early stopping (patience=10)
```

**文件**: `trainer.py` — `trainer_custom()`

### 5.3 优化器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Optimizer | AdamW | 权重衰减正则化 |
| base_lr | 0.001 | 学习率 |
| weight_decay | 0.1 | 权重衰减系数 |
| betas | (0.9, 0.999) | Adam 动量参数 |
| lr_exp | 0.9 | 多项式衰减指数 |

**文件**: `trainer.py:108-111`

### 5.4 模型保存与加载

```python
# 保存 (仅保存可训练参数)
model.save_lora_parameters("best_model.pth")
# 包含: LoRA A/B 权重 + prompt_encoder state_dict + mask_decoder state_dict

# 加载
net = LoRA_Sam(sam, args.rank)
net.load_lora_parameters("best_model.pth")
```

**文件**: `sam_lora_image_encoder.py` — `LoRA_Sam.save_lora_parameters()` / `load_lora_parameters()`

---

## 6. 推理系统

### 6.1 批量推理（测试集）

```bash
python infer.py \
    --lora_ckpt models/best_model.pth \
    --img_size 1024 --num_classes 1 --rank 4 \
    --label_dir data/VOCdevkit/VOC2012/GD_Label
    --text_prompt "waterbody" \
    --gd_ckpt checkpoints/groundingdino_swinb_cogcoor.pth \
    --gd_config GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py
```

**流程**:
1. 遍历 test set (98 images)
2. 可选择使用 GD 在线生成框 or 读取预生成框
3. LoRA-SAM 前向推理
4. Softmax → ArgMax 得到分割结果
5. 计算 mIoU, mAcc, oAcc, Precision, Recall, F1

**文件**: `infer.py`

### 6.2 单图推理 + 可视化

```bash
python infer_one.py \
    --image_path path/to/image.jpg \
    --lora_ckpt models/best_model.pth \
    --img_size 1024 --num_classes 1 --rank 4 \
    --text_prompt "waterbody" \
    --gd_ckpt checkpoints/groundingdino_swinb_cogcoor.pth \
    --gd_config GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py
```

**输出** (3 种格式):
1. 分割掩码 PNG（单通道，0=背景, 255=水体）
2. 可视化图（3 面板：原图+框 / 预测叠加 / 置信度热图）
3. 控制台日志（类别、面积等信息）

**文件**: `infer_one.py`

---

## 7. 评估体系

### 7.1 评估指标

基于混淆矩阵计算以下指标（对每个类别和整体）：

| 指标 | 公式 | 说明 |
|------|------|------|
| **IoU** | `TP / (TP + FP + FN)` | 交并比 |
| **mIoU** | `mean(IoU_per_class)` | 平均交并比 |
| **Accuracy** | `TP / (TP + FN)` | 像素准确率 |
| **mAcc** | `mean(Acc_per_class)` | 平均准确率 |
| **oAcc** | `ΣTP / Σ(TP+FP+FN)` | 全局准确率 |
| **Precision** | `TP / (TP + FP)` | 精确率 |
| **Recall** | `TP / (TP + FN)` | 召回率 |
| **F1** | `2×P×R / (P+R)` | F1 分数 |

**文件**: `eval_metrics.py` — `mean_iou()`, `intersect_and_union()`

### 7.2 典型结果

| 类别 | IoU | Accuracy |
|------|-----|----------|
| Background | 90.26% | 95.32% |
| Waterbody | 76.23% | 83.50% |
| **mIoU** | **83.25%** | — |

（基于 Swin-B GD + LoRA-SAM, `r=4`, 50 epochs）

---

## 8. 代码模块索引

### 8.1 RiverSeg 仓库 (`E:\work\seg\river_seg`)

| 文件 | 功能 | 关键类/函数 |
|------|------|-------------|
| `train.py` | 训练入口 | 参数解析, LoRA-SAM 构建 |
| `trainer.py` | 训练循环 | `trainer_custom()`, `calc_loss_multiclass()` |
| `infer.py` | 批量推理 | `inference()` |
| `infer_one.py` | 单图推理 | `build_model()`, `predict()`, `visualize()` |
| `sam_lora_image_encoder.py` | LoRA-SAM 模型定义 | `LoRA_Sam`, `_LoRA_qkv` |
| `grounding_dino_wrapper.py` | GD 包装器 | `GroundingDINOWrapper`, `generate_boxes()` |
| `precompute_gd_boxes.py` | 离线生成 GD 框 | 批量处理数据集 |
| `eval_metrics.py` | 评估指标 | `mean_iou()`, `intersect_and_union()` |
| `utils.py` | 损失函数 | `DiceLoss`, `Focal_loss`, `BinaryDiceLoss` |
| `datasets/dataset_custom.py` | 数据集 | `MultiClassVOCSegmentation` |
| `datasets/transforms.py` | 数据变换 | `Resize`, `RandomHorizontalFlip` 等 |

### 8.2 GroundingDINO 仓库 (`E:\work\seg\GroundingDINO`)

| 文件 | 功能 |
|------|------|
| `groundingdino/models/GroundingDINO/groundingdino.py` | 主模型 `GroundingDINO` 类, `build_groundingdino()` |
| `groundingdino/models/GroundingDINO/transformer.py` | Transformer 编码器/解码器 |
| `groundingdino/models/GroundingDINO/bertwarper.py` | BERT 文本编码器包装 |
| `groundingdino/models/GroundingDINO/fuse_modules.py` | 图像-文本双向注意力融合 |
| `groundingdino/models/GroundingDINO/backbone/backbone.py` | 骨干网络 (ResNet/Swin) |
| `groundingdino/models/GroundingDINO/backbone/swin_transformer.py` | Swin Transformer |
| `groundingdino/models/GroundingDINO/backbone/position_encoding.py` | 位置编码 |
| `groundingdino/models/GroundingDINO/ms_deform_attn.py` | 多尺度可变形注意力 |
| `groundingdino/models/GroundingDINO/utils.py` | MLP, ContrastiveEmbed, Focal loss |
| `groundingdino/config/GroundingDINO_SwinB_cfg.py` | Swin-B 配置 |
| `groundingdino/config/GroundingDINO_SwinT_OGC.py` | Swin-T 配置 |
| `groundingdino/util/inference.py` | 推理 API (`load_model`, `predict`) |

---

## 9. 运行环境

### 9.1 依赖

```txt
# requirements.txt
torch>=1.10.0
torchvision
numpy
Pillow
tqdm
matplotlib
```

### 9.2 外部依赖

| 依赖 | 来源 | 说明 |
|------|------|------|
| MobileSAM | 本地 `MobileSAM/` | SAM 模型实现（`sam_model_registry`） |
| GroundingDINO | `E:\work\seg\GroundingDINO` | 开放词汇检测（需 `pip install -e .`） |

### 9.3 模型权重

| 模型 | 路径 | 大小 |
|------|------|------|
| SAM ViT-B | `checkpoints/sam_vit_b_01ec64.pth` | ~375MB |
| GroundingDINO Swin-B | `checkpoints/groundingdino_swinb_cogcoor.pth` | ~1.5GB |
| LoRA 微调权重 | `models/best_model_lora.pth` | ~15MB |

---

## 10. 架构决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| 检测模型 | YOLO / GroundingDINO | GroundingDINO | 开放词汇，无需预定义类别集 |
| 分割模型 | DeepLab / SAM / UNet | SAM | 强大的视觉基础模型，泛化能力强 |
| 微调方法 | Full FT / Adapter / LoRA | LoRA | 参数量仅6%，15MB权重，过拟合风险低 |
| LoRA 注入位置 | q/k/v / q+v / 全部 | **q+v** | 经验最优（原论文和实验验证） |
| LoRA rank | 1/2/4/8/16 | **4** | 平衡表达力与参数量 |
| 损失函数 | CE+Dice / Focal+Dice | Focal+Dice | 缓解类别不均衡 |
| 框来源 | 手动标注 / GD预生成 / 在线GD | **两者兼有** | 训练用预生成保证速度，推理用在线保证灵活 |
| 图像尺寸 | 512/768/1024/1536 | **1024** | SAM 默认输入尺寸 |
| 文本编码器 | CLIP / BERT | BERT-base | GD 原生设计 |

---

## 11. 扩展指南

### 11.1 添加新类别

1. 修改 `dataset_custom.py` 的 `get_dataset_metainfo()` 中的类别列表和调色板
2. 运行 `precompute_gd_boxes.py --text_prompt "new category"`
3. 训练时设置 `--num_classes N`

### 11.2 更换检测模型

1. 实现新的 Wrapper 类（参照 `GroundingDINOWrapper` 接口）
2. 确保提供 `generate_boxes(image, text_prompt)` 方法
3. 框坐标格式：`[x1, y1, x2, y2]`（原始图像像素坐标）

### 11.3 更换分割骨干

修改 `train.py` 中的 `vit_name`：
- `vit_b`: ViT-Base (默认，768 dim, 12 layers)
- `vit_h`: ViT-Huge (1280 dim, 32 layers)
- `vit_t`: TinyViT (轻量 MobileSAM)
