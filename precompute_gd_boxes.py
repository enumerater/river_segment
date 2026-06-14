"""
Pre-compute Grounding DINO bounding boxes offline for faster training.

Iterates over all images in the VOC dataset, runs GroundingDINO with
a text prompt, and saves the resulting boxes as TxtLabel-compatible .txt files.

Usage:
    python precompute_gd_boxes.py \
        --gd_ckpt checkpoints/groundingdino_swinb_cogcoor.pth \
        --gd_config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py \
        --text_prompt "waterbody" \
        --root_path data/VOCdevkit/VOC2012 \
        --output_dir data/VOCdevkit/VOC2012/GD_Label
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from grounding_dino_wrapper import GroundingDINOWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def read_split(root_path, split_name):
    """Read image names from a split file (train.txt / val.txt / test.txt)."""
    split_file = os.path.join(root_path, "ImageSets", "Segmentation", split_name)
    if not os.path.exists(split_file):
        logging.warning(f"Split file not found: {split_file}")
        return []
    with open(split_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute Grounding DINO boxes for VOC dataset"
    )
    parser.add_argument("--gd_ckpt", type=str, required=True,
                        help="Path to Grounding DINO checkpoint")
    parser.add_argument("--gd_config", type=str, required=True,
                        help="Path to Grounding DINO config file")
    parser.add_argument("--text_prompt", type=str, required=True,
                        help='Text prompt (e.g. "waterbody")')
    parser.add_argument("--root_path", type=str,
                        default="data/VOCdevkit/VOC2012",
                        help="VOC dataset root directory")
    parser.add_argument("--output_dir", type=str,
                        default="data/VOCdevkit/VOC2012/GD_Label",
                        help="Output directory for generated box files")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train.txt", "val.txt", "test.txt"],
                        help="Split files to process")
    parser.add_argument("--box_threshold", type=float, default=0.25,
                        help="Box confidence threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="Text confidence threshold")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run GD on")

    args = parser.parse_args()

    img_dir = os.path.join(args.root_path, "JPEGImages")
    if not os.path.exists(img_dir):
        logging.error(f"Image directory not found: {img_dir}")
        sys.exit(1)

    # Collect all image names from requested splits
    all_names = []
    for split in args.splits:
        names = read_split(args.root_path, split)
        logging.info(f"Found {len(names)} images in {split}")
        all_names.extend(names)

    # Deduplicate while preserving order
    seen = set()
    unique_names = []
    for name in all_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    logging.info(f"Total unique images to process: {len(unique_names)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Grounding DINO
    logging.info("Loading Grounding DINO...")
    wrapper = GroundingDINOWrapper(
        gd_ckpt_path=args.gd_ckpt,
        gd_config_path=args.gd_config,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    logging.info("Grounding DINO loaded successfully")

    # Process each image
    success = 0
    skipped = 0
    for name in tqdm(unique_names, desc="Generating boxes"):
        out_path = os.path.join(args.output_dir, f"{name}.txt")
        if os.path.exists(out_path):
            skipped += 1
            continue

        img_path = os.path.join(img_dir, f"{name}.jpg")
        if not os.path.exists(img_path):
            logging.warning(f"Image not found: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            boxes = wrapper.generate_boxes(image, args.text_prompt)

            # Save in TxtLabel format: each row has 4 comma-separated floats
            with open(out_path, "w") as f:
                for i in range(boxes.shape[0]):
                    x1, y1, x2, y2 = boxes[i].tolist()
                    f.write(f"{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}\n")

            success += 1
        except Exception as e:
            logging.error(f"Failed to process {name}: {e}")

    logging.info(
        f"Done! Generated {success} new files, {skipped} already existed"
    )


if __name__ == "__main__":
    main()