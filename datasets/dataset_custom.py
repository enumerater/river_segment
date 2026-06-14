import os
import torch.utils.data as data
import numpy as np
import torch
from datasets import transforms
from PIL import Image
import matplotlib.pyplot as plt


class MultiClassVOCSegmentation(data.Dataset):
    def __init__(self, args_img_size, img_path, gt_path, txt_file, train_val='train', base_size=None, crop_size=None, flip_prob=None, label_dir='data/VOCdevkit/VOC2012/TxtLabel'):
        super(MultiClassVOCSegmentation, self).__init__()

        if train_val == 'train':
            with open(txt_file, 'r') as f:
                data = [data.strip() for data in f.readlines() if len(data.strip()) > 0]
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=[args_img_size,args_img_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
            )
        else:
            with open(txt_file, 'r') as f:
                data = [data.strip() for data in f.readlines() if len(data.strip()) > 0]
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=[args_img_size,args_img_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
            )

        self.img_files = [os.path.join(img_path, i + '.jpg') for i in data]
        self.gt_files = [os.path.join(gt_path, i + '.png') for i in data]
        self.txt_files = [os.path.join(label_dir, i + '.txt') for i in data]

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        target = Image.open(self.gt_files[index])
        with open(self.txt_files[index], 'r') as file:
            values_str = file.read().strip()
            values = [float(v) for v in values_str.split(',')]
        tensor_data = torch.tensor([values], device='cpu')
        img, target = self.transforms(img, target)
        return img, target ,tensor_data

    def __len__(self):
        return len(self.img_files)
    
    def get_dataset_metainfo(self):
        metainfo = dict(
            classes=[
                'BG [Background]',
                'WB [WaterBody]',

            ],
            palette=[
                [0, 0, 0],
                [255, 255, 255]
            ]
        )
#         metainfo = dict(
#             classes=[
#                 'Background', 
#                 'Ship hull', 
#                 'Propeller', 
#                 'Bilge keel', 
#                 'Anode',                     
#                 'Sea chest grating', 
#                 'Overboard valve', 
#                 'Corrosion', 
#                 'Paint peel', 
#                 'Marine growth', 
#                 'Defect'
#             ],
#             palette=[
#                 [0, 0, 0], 
#                 [0, 0, 255], 
#                 [128, 0, 128], 
#                 [255, 165, 0], 
#                 [0, 255, 255], 
#                 [255, 255, 255], 
#                 [64, 224, 208], 
#                 [255, 255, 0], 
#                 [255, 0, 0], 
#                 [0, 128, 0], 
#                 [255, 192, 203]
#             ]
#         )
        return metainfo

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

    
class BinaClassVOCSegmentation(data.Dataset):
    def __init__(self, args_img_size, img_path, gt_path, txt_file, train_val='train', base_size=None, crop_size=None, flip_prob=None):
        super(BinaClassVOCSegmentation, self).__init__()

        if train_val == 'train':
            with open(txt_file, 'r') as f:
                data = [data.strip() for data in f.readlines() if len(data.strip()) > 0]
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=[args_img_size,args_img_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
            )
        else:
            with open(txt_file, 'r') as f:
                data = [data.strip() for data in f.readlines() if len(data.strip()) > 0]
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=[args_img_size,args_img_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
            )

        self.img_files = [os.path.join(img_path, i + '.jpg') for i in data]
        self.gt_files = [os.path.join(gt_path, i + '.bmp') for i in data]

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        target = Image.open(self.gt_files[index])
        img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_files)
    
    def get_dataset_metainfo(self):
        metainfo = dict(
            classes=[
                'Background', 
                'Foreground'
            ],
            palette=[
                [0, 0, 0], 
                [255, 255, 255]
            ]
        )
        
        return metainfo

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets    


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
