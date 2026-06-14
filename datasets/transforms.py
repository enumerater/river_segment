import numpy as np
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from scipy.ndimage.interpolation import zoom

def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, [0, 0, padw, padh], fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 将image 和 target 的短边缩放到size大小
        image = F.resize(image, [size])
        target = F.resize(target, [size], interpolation=transforms.InterpolationMode.NEAREST)
        return image, target

    
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # 将image 和 target 的短边缩放到size大小
        image = F.resize(image, [self.size[0],self.size[1]])
        target = F.resize(target, [self.size[0],self.size[1]], interpolation=transforms.InterpolationMode.NEAREST)
        return image, target

    
class GTDownSample(object):
    def __init__(self, low_res):
        self.low_res = low_res
    def __call__(self, image, target):
        image = np.array(image)
        target = np.array(target)
        label_h, label_w = target.shape # (256,256)
        target = zoom(target, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        return image, target
    
    
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = transforms.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
    
