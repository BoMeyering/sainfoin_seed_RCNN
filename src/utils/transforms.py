# transforms.py
# Defines all image augmentations for training and validation loops

import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transforms():
    """
    Parameters:
        None
    Returns:
        An Albumentations compose function for training imageset
    """
    return A.Compose([
        A.RandomBrightnessContrast(p=0.4),
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.MotionBlur(p=0.2),
        A.Blur(p=0.2, blur_limit=3),
        ToTensorV2(p=1.0)
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def val_transforms():
    """
    Parameters:
        None
    Returns:
        An Albumentations compose function for the validation imageset
    """
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def inf_transforms():
    return A.Compose([
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
