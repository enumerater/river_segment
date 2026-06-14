"""
Single-image inference & visualization for SAM-based segmentation.

Usage:
    python infer_one.py --image_path path/to/image.jpg --lora_ckpt best_model.pth
                         --img_size 1024 --adapt_sam_type 0 --num_classes 1
"""
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from importlib import import_module
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision.transforms import functional as TF

from learnable_prompt_sam import LearnablePromptSAM
from MobileSAM.mobile_sam import sam_model_registry


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


def build_model(args):
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    low_res = img_embedding_size * 4

    if args.adapt_sam_type == 0:
        net = sam.cuda()
        net.load_state_dict(torch.load(args.lora_ckpt))
    elif args.adapt_sam_type == 1:
        pkg = import_module(args.module)
        net = pkg.LoRA_Sam(sam, args.rank).cuda()
        assert args.lora_ckpt is not None
        net.load_lora_parameters(args.lora_ckpt)
    elif args.adapt_sam_type == 2:
        sam = sam.cuda()
        net = LearnablePromptSAM(sam=sam, num_classes=args.num_classes + 1)
        net = net.cuda()
        net.load_state_dict(torch.load(args.lora_ckpt))
    else:
        raise ValueError(f'Unknown adapt_sam_type: {args.adapt_sam_type}')

    net.eval()
    return net, low_res


def predict(net, image, prompt, multimask_output, img_size, adapt_sam_type):
    with torch.no_grad():
        if adapt_sam_type == 2:
            output_masks = net(image)
        else:
            outputs = net(image, multimask_output, img_size, prompt)
            output_masks = outputs['masks']
        probs = torch.softmax(output_masks, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0)
    return pred.cpu().numpy(), probs.cpu()


def visualize(image_tensor, pred_mask, probs, orig_size, save_path):
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
    axes[0].imshow(img_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    overlay = img_np.copy()
    overlay[pred_resized == 1] = [0, 0.8, 0]
    axes[1].imshow(overlay)
    axes[1].set_title('Prediction Overlay')
    axes[1].axis('off')

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
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder', help='LoRA module')
    parser.add_argument('--adapt_sam_type', type=int, default=0,
                        help='0: Full_finetune; 1: LoRA; 2: LearnablePrompt (no box)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Bounding box prompt (N*4 values). Default: full-image box. Ignored for type 2.')
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

    if args.prompt is not None:
        prompt = parse_prompt(args.prompt)
    else:
        prompt = get_default_prompt(args.img_size)
    prompt = prompt.unsqueeze(0).cuda()

    print(f'Building model (adapt_sam_type={args.adapt_sam_type})...')
    net, low_res = build_model(args)

    print(f'Loading image: {args.image_path}')
    image_tensor, orig_size = load_image(args.image_path, args.img_size)
    image_tensor = image_tensor.cuda()

    multimask_output = args.num_classes > 1
    pred_mask, probs = predict(net, image_tensor, prompt, multimask_output, args.img_size, args.adapt_sam_type)

    out_name = os.path.splitext(os.path.basename(args.image_path))[0]
    mask_path = os.path.join(args.save_dir, f'{out_name}_mask.png')
    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    mask_img.save(mask_path)
    print(f'Mask saved to {mask_path}')

    vis_path = os.path.join(args.save_dir, f'{out_name}_vis.png')
    visualize(image_tensor, pred_mask, probs, orig_size, vis_path)


if __name__ == '__main__':
    main()