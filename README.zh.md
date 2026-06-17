# SAMUSS — 基于SAM的语义分割

对 SAM（Segment Anything Model）进行 LoRA 微调，用于语义分割。

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
# LoRA 微调
python train.py --batch_size 1 --max_epochs 10 --img_size 1024 --num_classes 1 --rank 4
```

### 批量测试

```bash
python infer.py --lora_ckpt E:\work\seg\samuss\models\best_model.pth --img_size 1024 --num_classes 1 --rank 4 --label_dir data/VOCdevkit/VOC2012/GD_Label --text_prompt "waterbody" --gd_ckpt checkpoints/groundingdino_swinb_cogcoor.pth --gd_config E:\work\seg\GroundingDINO\groundingdino\config\GroundingDINO_SwinB_cfg.py
```

### 单图推理 + 可视化

```bash
python infer_one.py --image_path data/VOCdevkit/VOC2012/JPEGImages/00256.jpg  --lora_ckpt E:\work\seg\samuss\models\best_model.pth --img_size 1024 --text_prompt "waterbody" --gd_ckpt checkpoints/groundingdino_swinb_cogcoor.pth --gd_config E:\work\seg\GroundingDINO\groundingdino\config\GroundingDINO_SwinB_cfg.py

```

## 模型

本项目使用 **LoRA**（低秩适配）微调 SAM。只有图像编码器中注意力层的 q/k/v 投影矩阵通过低秩分解进行适配，提示编码器和掩码解码器则进行全参微调。权重通过 `model.save_lora_parameters()` / `model.load_lora_parameters()` 保存和加载。