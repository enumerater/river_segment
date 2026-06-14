# SAMUSS — 基于SAM的语义分割

对 SAM（Segment Anything Model）进行微调，用于语义分割。支持全参数微调和 LoRA 微调。

## 项目结构

```
.
├── train.py                       # 训练入口
├── infer.py                       # 批量推理+评估
├── infer_one.py                   # 单图推理+可视化
├── trainer.py                     # 训练循环 & 损失函数
├── sam_lora_image_encoder.py      # LoRA SAM 模型
├── eval_metrics.py                # 评估指标（mIoU, mAcc）
├── utils.py                       # 损失函数（Dice, Focal）
├── convert_checkpoint.py          # 权重格式转换
├── datasets/                      # 数据集加载 & 数据增强
│   ├── dataset_custom.py          # VOC 多类 / 二类数据集
│   └── transforms.py              # 图像变换
├── data/                          # 数据集（VOCdevkit）
├── checkpoints/                   # 预训练 SAM 权重
├── output/                        # 训练输出
├── MobileSAM/                     # SAM 模型实现
└── requirements.txt               # 依赖包
```

## 脚本说明

### 训练

```bash
# 全量微调
python train.py --adapt_sam_type 0 --batch_size 1 --max_epochs 10 --img_size 1024 --num_classes 1

# LoRA
python train.py --adapt_sam_type 1 --batch_size 1 --max_epochs 10 --img_size 1024 --num_classes 1 --rank 4
```

### 批量测试

```bash
python infer.py --lora_ckpt best_model.pth --adapt_sam_type 0 --img_size 1024 --num_classes 1
```

### 单图推理 + 可视化

```bash
python infer_one.py --image_path path/to/image.jpg --lora_ckpt best_model.pth --adapt_sam_type 0 --img_size 1024
```

## 模型类型

| Type | 模式 | 加载方式 |
|------|------|---------|
| 0 | Full Fine-tune | `model.load_state_dict()` |
| 1 | LoRA | `model.load_lora_parameters()` |