# SAMUSS — SAM-based Segmentation

Fine-tuning SAM (Segment Anything Model) for semantic segmentation. Supports full fine-tune and LoRA.

## Project Structure

```
.
├── train.py                       # Training entry
├── infer.py                       # Batch inference & evaluation
├── infer_one.py                   # Single-image inference + visualization
├── trainer.py                     # Training loop & loss functions
├── sam_lora_image_encoder.py      # LoRA SAM model
├── eval_metrics.py                # Evaluation metrics (mIoU, mAcc)
├── utils.py                       # Loss functions (Dice, Focal)
├── convert_checkpoint.py          # Checkpoint format converter
├── datasets/                      # Dataset loaders & transforms
│   ├── dataset_custom.py          # VOC multi-class / binary dataset
│   └── transforms.py              # Image transforms
├── data/                          # Dataset (VOCdevkit)
├── checkpoints/                   # Pretrained SAM weights
├── output/                        # Training outputs
├── MobileSAM/                     # SAM model implementation
└── requirements.txt               # Dependencies
```

## Scripts

### Train

```bash
# Full fine-tune
python train.py --adapt_sam_type 0 --batch_size 1 --max_epochs 10 --img_size 1024 --num_classes 1

# LoRA
python train.py --adapt_sam_type 1 --batch_size 1 --max_epochs 10 --img_size 1024 --num_classes 1 --rank 4
```

### Batch Test

```bash
python infer.py --lora_ckpt best_model.pth --adapt_sam_type 0 --img_size 1024 --num_classes 1
```

### Single-Image Inference + Visualization

```bash
python infer_one.py --image_path path/to/image.jpg --lora_ckpt best_model.pth --adapt_sam_type 0 --img_size 1024
```

## Model Types

| Type | Mode | Description | Need Box |
|------|------|-------------|----------|
| 0 | Full Fine-tune | `model.load_state_dict()` | Yes |
| 1 | LoRA | `model.load_lora_parameters()` | Yes |
| 2 | Learnable Prompt | `model.load_state_dict()` | **No** |