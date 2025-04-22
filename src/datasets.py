"""
src/datasets.py
Dataset classes
BoMeyering 2025
"""

import torch
import cv2
from glob import glob
from typing import Union
from pathlib import Path
from pycocotools.coco import COCO

from torch.utils.data import Dataset

class SeedDataset(Dataset):
    def __init__(self, image_dir: Union[Path, str], label_dir: Union[Path, str]):
        self.img_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_names = [name for name in glob("*", root_dir=self.img_dir) if name.lower().endswith(('jpg', 'jpeg'))]
        self.coco_raw = COCO(self.label_dir)
        
        self.mapping = {}

        for img_id, annotations in self.coco_raw.imgToAnns.items():
            file_name = self.coco_raw.imgs[img_id]['file_name']
            if file_name in self.img_names:
                self.mapping[img_id] = {
                    'file_name': file_name,
                    'path': Path(self.img_dir) / file_name
                }
        self.img_ids = list(self.mapping.keys())



    def __getitem__(self, index: int):
        img_id = self.img_ids[index]
        img = cv2.imread(self.mapping[img_id]['path'])

        return img
    
    def __len__(self):
        return len(self.image_names)