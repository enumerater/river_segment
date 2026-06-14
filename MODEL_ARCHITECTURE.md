# SAMUSS 模型架构文档

## 项目概述

SAMUSS (SAM-based Universal Semantic Segmentation) 是基于 Segment Anything Model (SAM) 的语义分割项目。支持全参数微调和 LoRA 微调两种策略。

---

## 1. 整体框架

```
输入图像 → SAM Image Encoder → 图像特征 → SAM Mask Decoder → 分割结果
                                        ↑
                                    Prompt Encoder
                                        ↑
                                   Bounding Box
```

- **Backbone**: SAM ViT-B (vit_b, 默认)
- **输入尺寸**: 1024×1024（可配置）
- **输出**: 与输入同分辨率的分割掩码

---

## 2. 模型架构详情

### 2.1 图像编码器 (ImageEncoderViT)

| 组件 | 参数 |
|------|------|
| Patch Embed | 16×16, 768 dim |
| Transformer Blocks | 12层 |
| Attention Heads | 12 |
| Hidden Dim | 768 |
| MLP Ratio | 4 |
| Neck | Conv2D(768→256, k=1) → LayerNorm2d(256) → Conv2D(256→256, k=3) → LayerNorm2d(256) |
| 输出尺寸 | 64×64×256 |

### 2.2 提示编码器 (PromptEncoder)
- 支持点、框、掩码三种提示
- 框提示: 位置编码 + 可学习嵌入
- 掩码下采样: Conv2D 序列 → 输出 256-d 特征

### 2.3 掩码解码器 (MaskDecoder)
- TwoWayTransformer: 2层, 256 dim, 8 heads, MLP 2048
- 上采样: ConvTranspose2D(256→64→32) + LayerNorm2d
- 输出头: Hypernetwork MLPs 生成掩码

---

## 3. LayerNorm2d 实现

项目中 `LayerNorm2d` 位于 `MobileSAM/mobile_sam/modeling/common.py`。

### 当前版本 (标准实现)
```python
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        self.weight = nn.Parameter(ones(num_channels))    # 形状 (C,)
        self.bias   = nn.Parameter(zeros(num_channels))   # 形状 (C,)

    def forward(self, x):
        # 通道维归一化: (x - mean) / sqrt(var + eps) * weight + bias
```

---

## 4. 微调策略 (adapt_sam_type)

| Type | 名称 | 说明 | 保存的权重 | 加载方式 |
|------|------|------|-----------|---------|
| 0 | Full Fine-tune | 全参数微调 | 完整 state_dict | `net.load_state_dict(ckpt)` |
| 1 | LoRA | 低秩适配 (rank=4) | LoRA + prompt/mask 头 | `net.load_lora_parameters(ckpt)` |

---

## 5. 数据流

```
输入图像 [B, 3, H, W]
    ↓ Normalize (ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ↓ Resize to [1024, 1024]
    ↓
SAM.preprocess: (x - pixel_mean) / pixel_std
    pixel_mean=[0,0,0], pixel_std=[1,1,1] → 恒等变换
    ↓
Image Encoder → [B, 256, 64, 64]
    ↓
Prompt Encoder (box: [0,0,1024,1024]) → sparse + dense embeddings
    ↓
Mask Decoder → low_res masks [B, num_classes+1, 256, 256]
    ↓ postprocess_masks: interpolate → [B, C, 1024, 1024]
    ↓
Softmax + Argmax → 最终掩码 [B, 1024, 1024]
```

---

## 6. 损失函数

- **Focal Loss**: alpha=0.25, gamma=2
- **Dice Loss**: 多分类 Dice (num_classes + 1)
- **组合**: `loss = 0.25 * focal + 0.75 * dice` (dice_param=0.75)

---

## 7. 常见问题排查

### 7.1 加载权重失败 "Missing/Unexpected keys"

**原因**: `common.py` 中 LayerNorm2d 版本与训练时不匹配。

**解决**:
- `best_model.pth` → 使用标准 LayerNorm2d
- `111.pth` → 运行 `python convert_checkpoint.py 111.pth 111_converted.pth` 转换

### 7.2 分割结果不完整

**可能原因**:
1. 模型训练不充分
2. 数据集标注质量或类别定义问题
3. 可调整 prompt 或后处理参数

### 7.3 正确推理命令

```bash
# Full Fine-tune (best_model.pth)
python infer_one.py --image_path path/to/image.jpg --lora_ckpt best_model.pth --adapt_sam_type 0 --img_size 1024

# LoRA (111.pth 需先转换)
python convert_checkpoint.py 111.pth 111_converted.pth
python infer_one.py --image_path path/to/image.jpg --lora_ckpt 111_converted.pth --adapt_sam_type 1 --img_size 1024
```

---

## 8. 文件结构

```
samuss/
├── train.py                          # 训练入口
├── infer.py                          # 批量推理+评估
├── infer_one.py                      # 单图推理
├── trainer.py                        # 训练循环
├── utils.py                          # 损失函数
├── eval_metrics.py                   # 评估指标
├── sam_lora_image_encoder.py         # LoRA 实现
├── convert_checkpoint.py             # 权重格式转换
├── MobileSAM/mobile_sam/
│   ├── build_sam.py                  # 模型构建入口
│   └── modeling/
│       ├── sam.py                    # SAM 主模型
│       ├── image_encoder.py          # ViT 图像编码器
│       ├── prompt_encoder.py         # 提示编码器
│       ├── mask_decoder.py           # 掩码解码器
│       ├── common.py                 # LayerNorm2d, MLPBlock
│       └── tiny_vit_sam.py           # TinyViT
└── datasets/
    ├── dataset_custom.py             # VOC 数据集
    └── transforms.py                 # 数据增强
```