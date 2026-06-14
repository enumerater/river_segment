import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datasets
from typing import Dict, Optional
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from eval_metrics import mean_iou
from datasets.dataset_custom import MultiClassVOCSegmentation, BinaClassVOCSegmentation

from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from utils import DiceLoss, Focal_loss, BinaryDiceLoss

from MobileSAM.mobile_sam import sam_model_registry


def calc_loss_multiclass(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def trainer_custom(args, model, snapshot_path, multimask_output, low_res):

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    root = 'data/VOCdevkit/VOC2012'
    img_path = os.path.join(root, 'JPEGImages')
    gt_path = os.path.join(root, 'SegmentationClass')
    train_txt = os.path.join(root, 'ImageSets/Segmentation/train.txt')
    val_txt = os.path.join(root, 'ImageSets/Segmentation/val.txt')
    test_txt = os.path.join(root, 'ImageSets/Segmentation/test.txt')

    assert os.path.exists(img_path), 'img_path not exists'
    assert os.path.exists(gt_path), 'gt_path not exists'
    assert os.path.exists(train_txt), 'train_txt not exists'
    assert os.path.exists(val_txt), 'val_txt not exists'
    assert os.path.exists(test_txt), 'test_txt not exists'

    if args.dataset == 'Custom_MultiClass':
        db_train = MultiClassVOCSegmentation(args.img_size, img_path, gt_path, train_txt, train_val='train')
        db_val = MultiClassVOCSegmentation(args.img_size, img_path, gt_path, val_txt, train_val='val')

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))
    meta_info = db_val.get_dataset_metainfo()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    if args.dataset == 'Custom_MultiClass':
        ce_loss = Focal_loss()
        dice_loss = DiceLoss(num_classes + 1)
    elif args.dataset == 'Custom_BinaClass':
        ce_loss = Focal_loss()
        dice_loss = DiceLoss(num_classes + 1)

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0
    best_miou=0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, prompt = sampled_batch
            image_batch, label_batch, prompt = image_batch.cuda(), label_batch.cuda(), prompt.cuda()

            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            outputs = model(image_batch, multimask_output, args.img_size, prompt)
            outputs = outputs['masks']

            if args.dataset == 'Custom_MultiClass':
                loss, loss_ce, loss_dice = calc_loss_multiclass(outputs, label_batch, ce_loss, dice_loss, args.dice_param)
            elif args.dataset == 'Custom_BinaClass':
                loss, loss_ce, loss_dice = calc_loss_multiclass(outputs, label_batch, ce_loss, dice_loss, args.dice_param)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

        prediction_list=[]
        gtlabel_list=[]

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in tqdm(enumerate(valloader)):
                image, label, prompt = sampled_batch
                image, label = image.cuda(), label.cuda()
                prompt = prompt.cuda()
                label = label.squeeze(0)
                gt = label.cpu().detach().numpy()
                with torch.no_grad():
                    outputs = model(image, multimask_output, args.img_size, prompt)
                    output_masks = outputs['masks']

                    out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                    prediction = out.cpu().detach().numpy()

                    prediction_list.append(prediction)
                    gtlabel_list.append(gt)

            eval_metrics = mean_iou(
                meta_info['classes'],
                prediction_list,
                gtlabel_list,
                num_labels=num_classes + 1,
                ignore_index=255
            )
            current_miou = eval_metrics["mean_iou"]
            current_macc = eval_metrics["mean_accuracy"]
            current_oacc = eval_metrics["overall_accuracy"]
            current_iou_per_cat = eval_metrics["per_category_iou"]
            current_acc_per_cat = eval_metrics["per_category_accuracy"]

        print('\n')
        logging.info(f'--------------------------------------- Evaluating Metrics ---------------------------------------')
        for cls_name, cls_acc, cls_iou in zip(meta_info['classes'], current_acc_per_cat, current_iou_per_cat):
            logging.info(f'{cls_name}:  acc:{np.round(np.nanmean(cls_acc) * 100, 2)}  iou:{np.round(np.nanmean(cls_iou) * 100, 2)}')
        logging.info(f'--------------------------------------------------------------------------------------------------')
        logging.info("Valing Finished! mIoU={}, mAcc={}, oAcc={}, epoch={}".format(
            np.round(np.nanmean(current_miou) * 100, 2), np.round(np.nanmean(current_macc) * 100, 2), np.round(np.nanmean(current_oacc) * 100, 2), epoch_num))
        logging.info(f'--------------------------------------------------------------------------------------------------')
        print('\n')

        model.train()

        if current_miou > best_miou:
            best_miou = current_miou
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            if args.adapt_sam_type == 0:
                torch.save(model.state_dict(), save_mode_path)
            else:
                try:
                    model.save_lora_parameters(save_mode_path)
                except:
                    model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        save_interval = args.save_weight_interval
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            if args.adapt_sam_type == 0:
                torch.save(model.state_dict(), save_mode_path)
            else:
                try:
                    model.save_lora_parameters(save_mode_path)
                except:
                    model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            if args.adapt_sam_type == 0:
                torch.save(model.state_dict(), save_mode_path)
            else:
                try:
                    model.save_lora_parameters(save_mode_path)
                except:
                    model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    return "Training Finished!"
