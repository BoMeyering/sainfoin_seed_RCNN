import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transforms():
    """
    Returns an Albumentations compose function for training imageset
    """

    return A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        # A.CenterCrop(width=1000, height=1000, min_visibility=.8, p = .5),
        # A.RandomSizedBBoxSafeCrop(height=1000, width=1000, erosion_rate=0, interpolation=1, p=1),
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
    Returns an Albumentations compose function for the validation imageset
    """

    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def collate_fn(batch):
    """
    To handle the data loading as different images may have a different number of objects
    and to handle varying size tensors as well
    :param batch:
    :return:
    """

    return tuple(zip(*batch))
