"""
Train SAM for segmentation with LoRA fine-tuning.

Usage:
    python train.py --batch_size 1 --max_epochs 10 --img_size 1024 --num_classes 1 --rank 4
"""
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from sam_lora_image_encoder import LoRA_Sam
from MobileSAM.mobile_sam import sam_model_registry
from trainer import trainer_custom

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/VOCdevkit/VOC2012', help='root dir for data')
parser.add_argument('--output', type=str, default='output/WGSD')
parser.add_argument('--dataset', type=str, default='Custom_MultiClass', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=10, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Resume from checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', default=False, help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--dice_param', type=float, default=0.75)
parser.add_argument('--lr_exp', type=float, default=0.9, help='The learning rate decay expotential')
parser.add_argument("--weight_decay", default=0.1, type=float, help="weight decay for the optimizer")
parser.add_argument('--save_weight_interval', type=int, default=1, help='the interval that save trained weight')
parser.add_argument('--label_dir', type=str, default='data/VOCdevkit/VOC2012/TxtLabel',
                    help='Directory with box label .txt files (default: TxtLabel, switch to GD_Label for GD boxes)')
args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    snapshot_path = snapshot_path + '_dice' + str(args.dice_param) if args.dice_param != 0.8 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    print(args.vit_name)
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt,
                                                                pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    print('Using LoRA!')
    net = LoRA_Sam(sam, args.rank).cuda()

    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer = {
        'Custom_MultiClass': trainer_custom,
    }

    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
