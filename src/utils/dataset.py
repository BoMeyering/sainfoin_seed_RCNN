from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from glob import glob
from sklearn.utils.random import sample_without_replacement
import numpy as np
import os
import pandas as pd
import cv2
import torch


def dir_sampler(dir: str, k: int) -> list:
    filenames = glob(f"{dir}/*")
    print(filenames)

    idx = sample_without_replacement(n_population=len(filenames), n_samples=k, random_state=123)
    print(idx)

    return [filenames[i].split('/')[-1] for i in idx]

class SeedDataset(Dataset):
  """
  Subclass of Dataset for sainfoin seed images and annotations
  """

  def __init__(self, image_dir: str, annotation_path: str, resize_dims: tuple[int, int]=None, transforms: A.Compose=None, subset: list=None):
    """
    Initiate the SeedDataset class
    Parameters:
      image_dir (str): Path to image directory
      annotation_path (str): Path to coco annotation file
      resize_dims (tuple[int, int]): A tuple of two integers representing the image resize dimensions
      transforms (A.Compose): Albumentations transform function
      subset: a list of img_ids to restrcit the dataset to within the given image_dir
    Returns:
      An instantiated SeedDataset object
    """

    self.image_paths = glob(image_dir+"/*")
    self.image_names = [i.split('/')[-1] for i in self.image_paths]
    self.resize_dims = resize_dims
    self.transforms = transforms
    self._coco_raw = COCO(annotation_path)
    self.anns = self._coco_raw.imgToAnns
    self.classes = [i for i in self._coco_raw.cats.values()]
    self.mapping = {}
    self.ids = []


    # Filter to image subset
    if subset is not None:
      for img_id in subset:
        try:
          img_name= self._coco_raw.imgs[img_id]['file_name']
          if img_name in self.image_names:
            self.mapping[img_id] = {
              'img_name': img_name,
              'img_path': self.image_paths[self.image_names.index(img_name)]
            }
            self.ids.append(img_id)
          else:
            print(f"Image {img_name} not in {image_dir}")
        except KeyError:
          print(KeyError(f"img_id {img_id} is not in the annotations file"))

    # Use all images (default)
    else:
      for img_id in self._coco_raw.imgs.keys():
        img_name= self._coco_raw.imgs[img_id]['file_name']
        if img_name in self.image_names:
          self.mapping[img_id] = {
            'img_name': img_name,
            'img_path': self.image_paths[self.image_names.index(img_name)]
          }
        else:
          continue
      self.ids = [img_id for img_id in self._coco_raw.getImgIds() if img_id in self.mapping.keys()]

  def __getitem__(self, idx):
    img_id = self.ids[idx]
    img_path = self.mapping[img_id]['img_path']
    img_name = self.mapping[img_id]['img_name']

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    img_shape = img.shape[:2]
    if self.resize_dims is not None:
      img_resized = cv2.resize(img, self.resize_dims)
    else: 
      img_resized = img
    img_resized /= 255.0

    if True:
      try:
        boxes = []
        labels = []
        targets = {
          'img_id': img_id,
          'boxes': None,
          'labels': None
        }
        anns_filtered = self.anns[img_id]
        for obj in anns_filtered:
          x1, y1 = obj['bbox'][0:2]
          x2, y2 = [a + b for a, b in zip([x1, y1], obj['bbox'][2:4])]
          if self.resize_dims:
            bbox = np.array([x1, y1, x2, y2]) / np.array(img_shape * 2)[::-1] * np.array(self.resize_dims)
          else:
            bbox = np.array([x1, y1, x2, y2])
          boxes.append(bbox)
          label = obj['category_id']
          labels.append(label)
      except:
        print("there was an exception")
      boxes = np.array(boxes)
      labels = np.array(labels).astype(np.int64)
      targets['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
      targets['labels'] = torch.as_tensor(labels, dtype=torch.int64)

      if self.transforms:
        transformed = self.transforms(
          image=img_resized,
          bboxes=targets['boxes'],
          labels=targets['labels']
        )
        img_resized = transformed['image']
        targets['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)

      return img_resized, targets
    else:
      print(f"Image {img_name} has no associated annotations")
      return None, None
    
  def __len__(self):
    return len(self.ids)
  
