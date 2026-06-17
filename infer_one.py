"""
Single-image inference & visualization for SAM-based LoRA segmentation.

Usage:
    # Default: full-image box prompt
    python infer_one.py --image_path path/to/image.jpg --lora_ckpt best_model.pth
                         --img_size 1024 --num_classes 1 --rank 4

    # Grounding DINO: text prompt generates bounding boxes automatically
    python infer_one.py --image_path path/to/image.jpg --lora_ckpt best_model.pth
                         --img_size 1024 --num_classes 1 --rank 4
                         --text_prompt "waterbody"
                         --gd_ckpt checkpoints/groundingdino_swinb_cogcoor.pth
                         --gd_config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py
"""
import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision.transforms import functional as TF

from sam_lora_image_encoder import LoRA_Sam
from MobileSAM.mobile_sam import sam_model_registry
from grounding_dino_wrapper import GroundingDINOWrapper


def get_default_prompt(img_size=1024):
    return torch.tensor([[0, 0, img_size, img_size]], dtype=torch.float32)


def parse_prompt(prompt_str):
    values = [float(v.strip()) for v in prompt_str.split(',')]
    if len(values) % 4 != 0:
        raise ValueError(f'Prompt must have N*4 values, got {len(values)}')
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 4)


def load_image(img_path, img_size):
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size
    img = TF.resize(img, [img_size, img_size])
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return img.unsqueeze(0), orig_size


def generate_gd_prompt(image_path, wrapper, text_prompt, img_size):
    """Run Grounding DINO on an image and return a SAM-compatible prompt tensor."""
    pil_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = pil_img.size
    boxes = wrapper.generate_boxes(pil_img, text_prompt)
    prompt = GroundingDINOWrapper.boxes_original_to_sam(boxes, (orig_w, orig_h), img_size)
    return prompt.unsqueeze(0)  # shape (1, N, 4) for SAM


def build_model(args):
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    low_res = img_embedding_size * 4

    net = LoRA_Sam(sam, args.rank).cuda()
    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    net.eval()
    return net, low_res


def predict(net, image, prompt, multimask_output, img_size):
    with torch.no_grad():
        outputs = net(image, multimask_output, img_size, prompt)
        output_masks = outputs['masks']
        probs = torch.softmax(output_masks, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0)
    return pred.cpu().numpy(), probs.cpu()


def visualize(image_tensor, pred_mask, probs, orig_size, save_path, boxes=None):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * std + mean, 0, 1)

    orig_w, orig_h = orig_size
    pred_img = Image.fromarray(pred_mask.astype(np.uint8) * 255)
    pred_img = pred_img.resize((orig_w, orig_h), Image.NEAREST)
    pred_resized = np.array(pred_img) // 255

    if probs.shape[1] > 1:
        conf_map = probs[0, 1].numpy()
    else:
        conf_map = probs[0, 0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Left: input image with bounding boxes ---
    axes[0].imshow(img_np)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor='red', linewidth=2, linestyle='--')
            axes[0].add_patch(rect)
    axes[0].set_title('Input Image + GD Boxes' if boxes is not None else 'Input Image')
    axes[0].axis('off')

    # --- Middle: prediction overlay with bounding boxes ---
    overlay = img_np.copy()
    overlay[pred_resized == 1] = [0, 0.8, 0]
    axes[1].imshow(overlay)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor='red', linewidth=2, linestyle='--')
            axes[1].add_patch(rect)
    axes[1].set_title('Prediction Overlay + GD Boxes' if boxes is not None else 'Prediction Overlay')
    axes[1].axis('off')

    # --- Right: confidence map ---
    axes[2].imshow(conf_map, cmap='viridis')
    axes[2].set_title('Foreground Confidence')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Visualization saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Single-image inference for SAM-based segmentation')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--save_dir', type=str, default='infer_results', help='Directory to save results')
    parser.add_argument('--img_size', type=int, default=1024, help='Input image size for the network')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of segmentation classes (excl. background)')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained SAM checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='best_model.pth', help='Fine-tuned checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='ViT model name')
    parser.add_argument('--rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Bounding box prompt (N*4 values). Default: full-image box.')
    # Grounding DINO arguments (mutually exclusive with --prompt)
    parser.add_argument('--text_prompt', type=str, default=None,
                        help='Text prompt for Grounding DINO (e.g. "waterbody"). Overrides --prompt when set.')
    parser.add_argument('--gd_ckpt', type=str, default=None,
                        help='Path to Grounding DINO checkpoint')
    parser.add_argument('--gd_config', type=str, default=None,
                        help='Path to Grounding DINO config file')
    parser.add_argument('--gd_box_threshold', type=float, default=0.25,
                        help='Box confidence threshold for GD')
    parser.add_argument('--gd_text_threshold', type=float, default=0.25,
                        help='Text confidence threshold for GD')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--deterministic', type=int, default=1, help='Use deterministic mode')
    args = parser.parse_args()

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    print('Building LoRA model...')
    net, low_res = build_model(args)

    print(f'Loading image: {args.image_path}')
    image_tensor, orig_size = load_image(args.image_path, args.img_size)
    image_tensor = image_tensor.cuda()

    # Determine prompt source: GD > explicit --prompt > default full-image box
    if args.text_prompt is not None:
        if args.gd_ckpt is None:
            print('Error: --gd_ckpt is required when using --text_prompt')
            sys.exit(1)
        if args.gd_config is None:
            print('Error: --gd_config is required when using --text_prompt')
            sys.exit(1)
        print(f'Generating boxes with Grounding DINO (prompt: "{args.text_prompt}")...')
        gd_wrapper = GroundingDINOWrapper(
            gd_ckpt_path=args.gd_ckpt,
            gd_config_path=args.gd_config,
            box_threshold=args.gd_box_threshold,
            text_threshold=args.gd_text_threshold,
        )
        prompt = generate_gd_prompt(args.image_path, gd_wrapper, args.text_prompt, args.img_size)
    elif args.prompt is not None:
        prompt = parse_prompt(args.prompt)
        prompt = prompt.unsqueeze(0)
    else:
        prompt = get_default_prompt(args.img_size)
        prompt = prompt.unsqueeze(0)
    prompt = prompt.cuda()

    multimask_output = args.num_classes > 1
    pred_mask, probs = predict(net, image_tensor, prompt, multimask_output, args.img_size)

    out_name = os.path.splitext(os.path.basename(args.image_path))[0]
    mask_path = os.path.join(args.save_dir, f'{out_name}_mask.png')
    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    mask_img.save(mask_path)
    print(f'Mask saved to {mask_path}')

    vis_path = os.path.join(args.save_dir, f'{out_name}_vis.png')
    # Extract boxes for visualization (SAM coordinates)
    boxes = prompt.squeeze(0).cpu().numpy() if prompt is not None and prompt.shape[1] > 0 else None
    visualize(image_tensor, pred_mask, probs, orig_size, vis_path, boxes=boxes)


if __name__ == '__main__':
    main()