import datetime
str(datetime.date.today())
import os
import sys
import torch


import pandas as pd
import numpy as np
import albumentations as A

from glob import glob
from tqdm import tqdm
from random import randint
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import ToTensor
from torchvision.ops import nms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
sys.path.append('./src')
from config import parse_config
from utils.loggers import create_logger
from model.model import create_model
from utils.transforms import train_transforms, val_transforms, collate_fn
from utils.dataset import SeedDataset, dir_sampler
from data_splitting import train, val, test
from model.train_val import train_model
from config import train_dir, val_dir, test_dir, annotation_dir, chkpt_dir, tensorboard_dir, log_dir
from config import device, cores, classes, n_classes, resize_to, n_epochs, batch_size
from config import base_name, lr, momentum, gamma

logger = create_logger()
logger.info('Training notebook started')

# os_details = !lsb_release -a
# os_details = '\n'.join(os_details)

# cpu_info = !lscpu
# cpu_info = '\n'.join(cpu_info)

# sys_info = 'Python ' + sys.version

# headings = ['OS_DETAILS:\n', 'CPU_INFO:\n', 'PYTHON KERNEL:\n']
# details = [os_details, cpu_info, sys_info]

# for h, d in zip(headings, details):
#   logger.info(h + d)


# Validate torch device settings
if device=='cuda':
  try:
    assert torch.cuda.is_available(), 'No CUDA device is available.'
    logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
    logger.info(f"CUDA device capabilities: {torch.cuda.get_device_capability()}")
    logger.info(f"CUDA device properties: {torch.cuda.get_device_properties(0)}")
  except AssertionError as e:
    logger.warning(e, exc_info=True)
    logger.info("Overwriting model config and setting device to 'cpu'.")
    device = torch.device('cpu')


# Set proportion sizes for subsampling the data
# sizes = [0.05, 0.1, 0.2, 0.5, 1.0]
# sizes = [0.05, 0.5, 1.0]
# sizes = [0.5, 1.0]
sizes = [1.0]
# sizes = [0.5]
# Read in all of the image metadata
img_data = pd.read_csv('./data/power_analysis/results.csv')
print(img_data.head())

img_data.loc[img_data['class']=='train'].shape

# Start Tensorboard writer
model_config = [
    "model_name:\tfasterrcnn",
    f"pretrained:\tTrue",
    f"classes:\t{classes}",
    f"n_classes:\t{n_classes}",
    f"lr:\t{lr}",
    f"momentum:\t{momentum}",
    f"n_epochs:\t{n_epochs}",
    f"batch_size:\t{batch_size}",
    f"lr_scheduler_gamma:\t{gamma}"
]
logger.info("MODEL_CONFIG\n"+"\n".join(model_config))

train_imgs = img_data.loc[img_data['class']=='train']
val_imgs = img_data.loc[img_data['class']=='val']

logger.info("Starting training scenarios.")
for size in sizes:
  today = str(datetime.date.today())
  writer = SummaryWriter(os.path.join(tensorboard_dir, f"{base_name}_{size}_{lr}_{today}"))
  logger.info("Starting Tensorboard Summary Writer.")
  logger.info(f"Scenario {sizes.index(size)+1}: training on imageset of {img_data.shape[0]*size}")
  logger.info(f"Creating train/validation splits for {size*100}% of the imageset in an 80/20 train/val split")
  if size < 1:
    train, _ = train_test_split(train_imgs,
                                test_size=size,
                                train_size=size,
                                stratify=train_imgs[['method']],
                                random_state=345)
    _, val = train_test_split(val_imgs,
                              test_size=size,
                              train_size=size,
                              stratify=val_imgs[['method']],
                              random_state=345)
  else:
    train = train_imgs.copy()
    val = val_imgs.copy()

  logger.info(f"Sampled images.\nTraining set: {train.shape[0]} images.\nVal set: {val.shape[0]} images")

  try:
    model_name = base_name + f"_{size}"
    model = create_model(n_classes, n_obj_det=500)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if not os.path.exists(f"./model_chkpt/{model_name}"):
      os.mkdir(f"./model_chkpt/{model_name}")
  except:
    logger.exception('An exception occurred.')

  # Create train dataset
  train_dataset = SeedDataset(image_dir=train_dir,
                              annot_dir=annotation_dir,
                              resize_dims=(resize_to, resize_to),
                              classes=classes,
                              transforms=train_transforms(),
                              subset=list(train['img_name'].unique()))

  # Create val dataset
  val_dataset = SeedDataset(image_dir=val_dir,
                            annot_dir=annotation_dir,
                            resize_dims=(resize_to, resize_to),
                            classes=classes,
                            transforms=val_transforms(),
                            subset=list(val['img_name'].unique()))

  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=cores,
    collate_fn=collate_fn,
  )

  val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=cores,
    collate_fn=collate_fn
  )

  train_model(model=model,
              optimizer=optimizer,
              scheduler=scheduler,
              n_epochs=n_epochs,
              device=device,
              train_loader=train_loader,
              val_loader=val_loader,
              logger=logger,
              writer=writer,
              model_name=model_name)

  writer.close()



















