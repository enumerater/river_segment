# SAMUSS — SAM-based Semantic Segmentation with LoRA

Fine-tuning [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) for **semantic segmentation** using **LoRA** (Low-Rank Adaptation). Supports both binary and multi-class segmentation on PASCAL VOC 2012 and custom datasets.

---

## Project Structure

```
.
├── train.py                       # Training entry point
├── infer.py                       # Batch inference & evaluation on test set
├── infer_one.py                   # Single-image inference + visualization
├── trainer.py                     # Training loop, loss functions, validation
├── sam_lora_image_encoder.py      # LoRA-wrapped SAM model
├── eval_metrics.py                # Evaluation metrics (mIoU, mAcc, precision, recall, F1)
├── utils.py                       # Loss functions (DiceLoss, Focal_loss)
├── precompute_gd_boxes.py         # Offline Grounding DINO box pre-computation
├── grounding_dino_wrapper.py      # Grounding DINO inference wrapper
├── datasets/                      # Dataset loaders & data transforms
│   ├── __init__.py
│   ├── dataset_custom.py          # VOC multi-class / binary dataset classes
│   └── transforms.py              # Image & mask transforms (Resize, Flip, Crop, Normalize)
├── data/                          # Dataset directory
│   └── VOCdevkit/VOC2012/         # PASCAL VOC 2012
│       ├── JPEGImages/            # 训练/验证/测试原图
│       ├── SegmentationClass/     # 语义分割标签 (PNG)
│       ├── ImageSets/Segmentation/ # 数据集划分 (train.txt, val.txt, test.txt)
│       ├── TxtLabel/              # 手工标注的包围盒 (TxtLabel 格式)
│       └── GD_Label/              # Grounding DINO 生成的包围盒
├── checkpoints/                   # Pretrained SAM weights
├── models/                        # Saved fine-tuned LoRA checkpoints
├── output/                        # Training & test outputs
│   ├── WGSD/                      # Training logs per experiment
│   └── test_results/              # Batch inference logs
├── vision/                        # Analysis & visualization scripts
│   ├── data_exploration.py        # Dataset distribution analysis
│   ├── training_curves.py         # Loss & metric curve plotting
│   ├── eval_analysis.py           # Confusion matrix & error analysis
│   └── augmentation_viz.py        # Data augmentation demo
├── MobileSAM/                     # MobileSAM model implementation
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── README.zh.md                   # 中文说明
```

## Environment

- **Python**: 3.9+
- **PyTorch**: 1.13+ / 2.x
- **CUDA**: 11.7+ (GPU required for training)

### Installation

```bash
# Clone repo
git clone <repo-url> && cd samuss

# Install dependencies
pip install -r requirements.txt

# Download SAM checkpoint
# Place at: checkpoints/sam_vit_b_01ec64.pth
# (vit_b recommended; also supports vit_h/vit_l/vit_t)

# (Optional) Download VOC 2012 dataset
# Place at: data/VOCdevkit/VOC2012/
```

## Dataset

This project uses **[PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)** with the waterbody binary segmentation task:

| Split | Samples | Description |
|-------|---------|-------------|
| Train | 683 | With data augmentation (random horizontal flip) |
| Val   | 195 | Validation for early stopping & model selection |
| Test  | 98  | Held-out test set for final evaluation |

**Data format:**
- Images: RGB JPEG (`data/VOCdevkit/VOC2012/JPEGImages/*.jpg`)
- Ground truth: PNG with class index per pixel (`SegmentationClass/*.png`)
  - `0` = Background, `1` = Waterbody
- Box prompts: TXT files with comma-separated bounding boxes (`TxtLabel/*.txt` or `GD_Label/*.txt`)
  - Each line: `x1,y1,x2,y2` (pixel coordinates)

**Data augmentation** applied during training:
- Resize to 1024×1024
- Random horizontal flip (p=0.5)
- Normalization (ImageNet mean/std)

## Model Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Input Image                       │
│                    (1024×1024×3)                     │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│              SAM Image Encoder (ViT-B)               │
│  ┌──────────────────────────────────────────────┐   │
│  │  Patch Embed → 12 Transformer Blocks → Neck   │   │
│  │  ┌──────────────────────────────────┐         │   │
│  │  │  LoRA: low-rank ΔW on q/k/v      │         │   │
│  │  │  W' = W + B·A    (rank r=4)      │         │   │
│  │  └──────────────────────────────────┘         │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  Bounding Box Prompt  ──►  Prompt Encoder           │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│              Mask Decoder (fine-tuned)               │
│         ┌──────────────────────────────┐             │
│         │  Cross-attention → MLP → up  │             │
│         └──────────────────────────────┘             │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│          Output Mask (H×W×num_classes)               │
│           Softmax → argmax → prediction              │
└─────────────────────────────────────────────────────┘
```

**Key design choices:**
- **LoRA (Low-Rank Adaptation):** Only the query/key/value projection matrices in the image encoder's self-attention are adapted via rank-4 decomposition. This avoids catastrophic forgetting and requires only ~0.5% additional parameters over full fine-tuning.
- **Frozen image encoder backbone:** Prevents overfitting on small datasets.
- **Fully fine-tuned mask decoder & prompt encoder:** These components are dataset-specific and need full adaptation.
- **Loss:** Focal loss (handles class imbalance) + Dice loss (optimizes IoU directly).

## Training

```bash
# Binary segmentation (waterbody)
python train.py \
    --batch_size 4 \
    --max_epochs 200 \
    --img_size 1024 \
    --num_classes 1 \
    --rank 4 \
    --base_lr 0.001 \
    --dice_param 0.75 \
    --AdamW
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_classes` | 1 | Number of foreground classes (excl. background) |
| `--batch_size` | 1 | Batch size |
| `--max_epochs` | 10 | Maximum training epochs |
| `--base_lr` | 0.001 | Initial learning rate |
| `--rank` | 4 | LoRA rank |
| `--dice_param` | 0.75 | Dice loss weight (loss = (1-w)*CE + w*Dice) |
| `--AdamW` | False | Use AdamW optimizer (faster convergence) |
| `--img_size` | 1024 | Input image resolution |
| `--label_dir` | `TxtLabel` | Box prompt directory (`GD_Label` for GD boxes) |

Training logs (loss, validation metrics) are written to `output/WGSD/<experiment_name>/log.txt`.

## Evaluation

### Batch test with metrics

```bash
# Manually labeled boxes
python infer.py \
    --lora_ckpt models/best_model.pth \
    --img_size 1024 \
    --num_classes 1 \
    --rank 4

# Grounding DINO auto box generation
python infer.py \
    --lora_ckpt models/best_model.pth \
    --img_size 1024 \
    --num_classes 1 \
    --rank 4 \
    --text_prompt "waterbody" \
    --gd_ckpt checkpoints/groundingdino_swinb_cogcoor.pth
```

### Single image inference

```bash
python infer_one.py \
    --image_path path/to/image.jpg \
    --lora_ckpt models/best_model.pth \
    --img_size 1024 \
    --num_classes 1
```

## Results

**Test set performance** (LoRA rank=4, SAM ViT-B, 200 epochs):

| Class | IoU (%) | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|-------|---------|-------------|--------------|-----------|-------|
| Background | 90.26 | 96.75 | 93.08 | 96.75 | 94.88 |
| Waterbody | 76.23 | 82.31 | 91.16 | 82.31 | 86.51 |
| **Overall** | **83.25** | **89.53** | **92.12** | **89.53** | **90.70** |

- **mIoU**: 83.25%
- **oAcc**: 92.58%
- **Best val mIoU during training**: 86.71% (epoch 25)

## Vision Analysis Scripts

```bash
# 1. Data exploration (class distribution, object sizes, sample viz)
python vision/data_exploration.py

# 2. Training curves (loss, validation metrics)
python vision/training_curves.py \
    --log_path output/WGSD/<experiment>/log.txt

# 3. Evaluation analysis (confusion matrix, per-class metrics)
python vision/eval_analysis.py \
    --log_path output/test_results/test_log/log.txt

# 4. Augmentation visualization
python vision/augmentation_viz.py
```

All figures saved to `vision/output/`.

## Visualizations

- **training_loss.png**: Training loss curves (total, CE, Dice) per iteration
- **validation_metrics.png**: Validation mIoU/mAcc/oAcc over epochs
- **per_class_iou.png**: Per-class IoU trends during training
- **confusion_matrix.png**: Confusion matrix (estimated from metrics)
- **per_class_metrics.png**: Per-class metric bar chart
- **overall_radar.png**: Radar chart of overall metrics
- **class_distribution.png**: Pixel-level class balance
- **object_size_distribution.png**: Foreground area ratio histogram
- **sample_grid.png**: Image + GT mask pairs
- **augmentation_grid.png**: Augmentation strategy comparison

## Citation

```bibtex
@article{kirillov2023sam,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and others},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and others},
  journal={arXiv:2106.09685},
  year={2021}
}
```
