"""
Batch inference & evaluation on test set.

Usage:
    # Full fine-tune model
    python infer.py --lora_ckpt best_model.pth --adapt_sam_type 0 --img_size 1024 --num_classes 1

    # LoRA model
    python infer.py --lora_ckpt best_model.pth --adapt_sam_type 1 --img_size 1024 --num_classes 1 --rank 4

    # Learnable Prompt (no box)
    python infer.py --lora_ckpt best_model.pth --adapt_sam_type 2 --img_size 1024 --num_classes 1
"""
import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from importlib import import_module
from tqdm.contrib import tzip

from learnable_prompt_sam import LearnablePromptSAM
from MobileSAM.mobile_sam import sam_model_registry

from PIL import Image

from eval_metrics import mean_iou
from datasets.dataset_custom import MultiClassVOCSegmentation, BinaClassVOCSegmentation


def inference(args, multimask_output, model):
    root = args.root_path
    img_path = os.path.join(root, 'JPEGImages')
    gt_path = os.path.join(root, 'SegmentationClass')
    test_txt = os.path.join(root, 'ImageSets/Segmentation/test.txt')

    assert os.path.exists(img_path), 'img_path not exists'
    assert os.path.exists(gt_path), 'gt_path not exists'
    assert os.path.exists(test_txt), 'test_txt not exists'

    if args.dataset == 'Custom_MultiClass':
        db_test = MultiClassVOCSegmentation(args.img_size, img_path, gt_path, test_txt, train_val='test')
    elif args.dataset == 'Custom_BinaClass':
        db_test = BinaClassVOCSegmentation(args.img_size, img_path, gt_path, test_txt, train_val='test')

    print("The length of test set is: {}".format(len(db_test)))

    with open(test_txt, 'r') as f:
        data = [data.strip() for data in f.readlines() if len(data.strip()) > 0]
    img_prefixs = [i for i in data]

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test iterations')
    meta_info = db_test.get_dataset_metainfo()

    prediction_list = []
    gtlabel_list = []
    model.eval()
    with torch.no_grad():
        for (i_batch, sampled_batch), img_prefix in tzip(enumerate(testloader), img_prefixs):
            image, label, prompt = sampled_batch
            image, label = image.cuda(), label.cuda()
            label = label.squeeze(0)
            gt = label.cpu().detach().numpy()

            if args.adapt_sam_type == 2:
                output_masks = model(image)
            else:
                prompt = prompt.cuda()
                outputs = model(image, multimask_output, args.img_size, prompt)
                output_masks = outputs['masks']

            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            prediction_list.append(prediction)
            gtlabel_list.append(gt)

        eval_metrics = mean_iou(meta_info['classes'], prediction_list, gtlabel_list, num_labels=args.num_classes + 1,
                                ignore_index=255)

        miou = eval_metrics["mean_iou"]
        macc = eval_metrics["mean_accuracy"]
        oacc = eval_metrics["overall_accuracy"]
        iou_per_cat = eval_metrics["per_category_iou"]
        acc_per_cat = eval_metrics["per_category_accuracy"]

    logging.info(f'--------------------------------------- Evaluating Metrics ---------------------------------------')
    for cls_name, cls_acc, cls_iou in zip(meta_info['classes'], acc_per_cat, iou_per_cat):
        logging.info(
            f'{cls_name}:  acc:{np.round(np.nanmean(cls_acc) * 100, 2)}  iou:{np.round(np.nanmean(cls_iou) * 100, 2)}')
    logging.info(f'--------------------------------------------------------------------------------------------------')
    logging.info("Testing Finished! mIoU={}, mAcc={}, oAcc={}".format(
        np.round(np.nanmean(miou) * 100, 2), np.round(np.nanmean(macc) * 100, 2), np.round(np.nanmean(oacc) * 100, 2)))
    logging.info(f'--------------------------------------------------------------------------------------------------')
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='data/VOCdevkit/VOC2012', help='root dir for data')
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--dataset', type=str, default='Custom_MultiClass', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='output/test_results')
    parser.add_argument('--img_size', type=int, default=1024, help='Input image size of the network')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, required=True, help='The fine-tuned checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    parser.add_argument('--adapt_sam_type', type=int, default=0,
                        help='0: Full_finetune; 1: LoRA; 2: LearnablePrompt (no box)')

    args = parser.parse_args()

    if args.config is not None:
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    if args.adapt_sam_type == 0:
        print('Using Full Fine-tune!')
        net = sam.cuda()
        net.load_state_dict(torch.load(args.lora_ckpt))
    elif args.adapt_sam_type == 1:
        print('Using LoRA!')
        pkg = import_module(args.module)
        net = pkg.LoRA_Sam(sam, args.rank).cuda()
        assert args.lora_ckpt is not None
        net.load_lora_parameters(args.lora_ckpt)
    elif args.adapt_sam_type == 2:
        print('Using Learnable Prompt SAM!')
        sam = sam.cuda()
        net = LearnablePromptSAM(sam=sam, num_classes=args.num_classes + 1)
        net = net.cuda()
        net.load_state_dict(torch.load(args.lora_ckpt))
    else:
        raise ValueError(f'Unknown adapt_sam_type: {args.adapt_sam_type}')

    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in net.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    inference(args, multimask_output, net)