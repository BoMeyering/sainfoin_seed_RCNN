from torch.utils.data import Dataset, DataLoader
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

    idx = sample_without_replacement(n_population=len(filenames), n_samples=k)
    print(idx)

    return [filenames[i].split('/')[-1] for i in idx]


class SeedDataset(Dataset):
  """
  Subclass of torch.utils.data.Dataset for the sainfoin seed classification problem
  """

  def __init__(self, image_dir, annot_dir, resize_dims, classes, transforms, subset):
    """
    Initiate the dataset with standard or custom params
    """
    self.labels = None
    self.dir_path = image_dir
    self.resize_dims = resize_dims
    self.classes = classes
    self.transforms = transforms
    self.subset = subset
    self.img_names = None
    self.img_paths = None

    if self.subset is not None:
      self.img_names = np.array(self.subset)
      self.img_paths = np.array([os.path.join(self.dir_path, i) for i in self.img_names])
    else:
      self.img_paths = np.array([i for i in glob(f"{self.dir_path}/*") if i.lower().endswith('.jpg')])
      self.img_names = np.array([img_path.split('/')[-1] for img_path in self.img_paths])

    sort_index = np.argsort(self.img_names)
    self.img_paths = self.img_paths[sort_index]
    self.img_names = self.img_names[sort_index]

    self.annot_paths = os.path.join(annot_dir, 'annotations_export.csv')
    self.annotations = pd.read_csv(self.annot_paths).drop(columns='Unnamed: 0')


  def __getitem__(self, idx):
    """
    """
    img_path = self.img_paths[idx]
    img_name = self.img_names[idx]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_shape = img.shape[:2]
    # img_resized = cv2.resize(img, self.resize_dims)
    img_resized = img
    img_resized /= 255.0


    if img_name in self.annotations.img_id.unique():
      boxes = []
      labels = []
      tmp_df = self.annotations.copy().loc[self.annotations.img_id==img_name]
      targets = {
          'img_name': img_name,
          }
      for row in tmp_df.values:
        bbox = list(row[5:])
        #scale and resize all bounding boxes according to self.resize_dims
        # bbox = np.array(bbox) / np.array(img_shape * 2)[::-1] * np.array(self.resize_dims*2)
        bbox = np.array(bbox)
        # print(bbox)
        boxes.append(bbox)
        label = list(self.classes.keys())[list(self.classes.values()).index(str(row[4]))]
        labels.append(label)


      boxes = np.array(boxes)
      labels = np.array(labels).astype(np.int64)
      targets['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
      targets['labels'] = torch.as_tensor(labels, dtype=torch.int64)
    else:
      print(f"image {img_name} not annotated)")
      return None, None

    if self.transforms:
      sample = self.transforms(
          image=img_resized,
          bboxes=targets['boxes'],
          labels=labels)
      img_resized = sample['image']
      targets['boxes'] = torch.Tensor(sample['bboxes'])
    else:
      img_resized = ToTensor(img_resized)

    return img_resized, targets

  def __len__(self):
    """
    """
    return len(self.img_names)




