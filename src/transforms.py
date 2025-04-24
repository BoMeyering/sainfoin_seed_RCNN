# transforms.py
# Defines all image augmentations for training and validation loops

import torch
from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(resize_dim: Tuple[int, int]=(3000, 3000)):
    """
    Parameters:
        None
    Returns:
        An Albumentations compose function for training imageset
    """
    return A.Compose([
        A.Normalize(normalization='min_max', max_pixel_value=255),
        # A.Resize(height=resize_dim[0], width=resize_dim[1]),
        A.RandomBrightnessContrast(p=0.4),
        A.SafeRotate(),
        A.HorizontalFlip(),
        A.MotionBlur(),
        A.Blur(p=0.2, blur_limit=3),
        A.Lambda(image=lambda x, **kwargs: x.astype('float32')),
        ToTensorV2(p=1.0)
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_val_transforms():
    """
    Parameters:
        None
    Returns:
        An Albumentations compose function for the validation imageset
    """
    return A.Compose([
        A.Normalize(normalization='min_max', max_pixel_value=255),
        A.Lambda(image=lambda x, **kwargs: x.astype('float32')),
        ToTensorV2(p=1.0)
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def get_inf_transforms():
    return A.Compose([
        A.Normalize(normalization='min_max', max_pixel_value=255),
        A.Lambda(image=lambda x, **kwargs: x.astype('float32')),
        ToTensorV2(p=1.0)
    ])


def collate_fn(batch):
    """
    Handles batches of varying sizes for the Dataloader
    Parameters:
        The batch of images passed from a Dataset in a Dataloader
    Returns:
        A zipped tuple of a given batch
    """
    return tuple(zip(*batch))
