# SAMUSS — SAM-based Segmentation

Fine-tuning SAM (Segment Anything Model) for semantic segmentation with LoRA.

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
# LoRA fine-tune
python train.py --batch_size 1 --max_epochs 10 --img_size 1024 --num_classes 1 --rank 4
```

### Batch Test

```bash
python infer.py --lora_ckpt best_model.pth --img_size 1024 --num_classes 1 --rank 4
```

### Single-Image Inference + Visualization

```bash
python infer_one.py --image_path path/to/image.jpg --lora_ckpt best_model.pth --img_size 1024 --num_classes 1 --rank 4
```

## Model

This project uses **LoRA** (Low-Rank Adaptation) to fine-tune SAM. Only the attention q/k/v projection matrices in the image encoder are adapted via low-rank decomposition, while the prompt encoder and mask decoder are fully fine-tuned. Checkpoints are saved/loaded with `model.save_lora_parameters()` / `model.load_lora_parameters()`.