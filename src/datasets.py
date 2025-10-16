"""
src/datasets.py
Dataset classes
BoMeyering 2025
"""

import torch
import cv2
import albumentations as A
import numpy as np
from copy import deepcopy
from sklearn.utils.random import sample_without_replacement
from glob import glob
from typing import Union
from pathlib import Path
from pycocotools.coco import COCO
from torch.utils.data import Dataset

def subset_sampler(img_ids: list[str], k: int) -> list:

    idx = sample_without_replacement(n_population=len(img_ids), n_samples=k, random_state=123)

    return [img_ids[i] for i in idx]

def ceil_div(a, b):
    return - (a // -b)

class SeedDataset(Dataset):
    def __init__(self, image_dir: Union[Path, str], label_dir: Union[Path, str], transforms: A.Compose, subset_size: float=1.0):
        self.img_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.subset_size = subset_size
        self.img_names = [name for name in glob("*", root_dir=self.img_dir) if name.lower().endswith(('jpg', 'jpeg'))]
        self.coco_raw = COCO(self.label_dir)

        # Create the img_id to img_name mapping
        self.mapping = {}
        for img_id, annotations in self.coco_raw.imgToAnns.items():
            file_name = self.coco_raw.imgs[img_id]['file_name']
            if file_name in self.img_names:
                self.mapping[img_id] = {
                    'file_name': file_name,
                    'path': Path(self.img_dir) / file_name
                }
        self.img_ids = list(self.mapping.keys())
        
        if self.subset_size < 1.0:
            k = round(len(self.img_ids) * subset_size)
            self.img_ids = subset_sampler(self.img_ids, k)


    def __getitem__(self, index: int):
        """ Get one training example by index """
        img_id = self.img_ids[index]
        img = cv2.imread(self.mapping[img_id]['path'], cv2.IMREAD_COLOR_RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            boxes = []
            labels = []
            targets = {
                'img_id': img_id,
                'boxes': None,
                'labels': None
            }
            filtered_anns = self.coco_raw.imgToAnns[img_id]
            for obj in filtered_anns:
                bbox = obj['bbox'][:2] + [a + b for a, b in zip(obj['bbox'][:2], obj['bbox'][2:])]
                boxes.append(np.array(bbox))
                labels.append(obj['category_id'])
        except Exception as e:
            print(e)
            return None, None
        targets['boxes'] = np.array(boxes).astype(np.float32)
        targets['labels'] = np.array(labels).astype(np.int64)

        transformed = self.transforms(
            image=img,
            bboxes=targets['boxes'],
            labels=targets['labels']
        )
        img = transformed['image']
        targets['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        targets['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)

        return img, targets
    
    def __len__(self):
        """ Get length of dataset """
        return len(self.img_ids)