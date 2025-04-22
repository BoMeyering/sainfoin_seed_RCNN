"""
train.py
Main training script
BoMeyering 2025
"""

import os
import torch
import datetime
import pandas as pd
import polars as pl
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision


# local script imports
from src.loggers import train_dir, val_dir, annotation_path
from src.loggers import tensorboard_dir
from src.loggers import device, cores, classes, n_classes
from src.loggers import n_epochs, batch_size, lr, momentum, gamma, base_name
from model.model import create_model
from model.train_val import train_model
from model.metrics import _iou_metrics, _map_metrics
from utils.dataset import SeedDataset
from utils.loggers import create_logger
from utils.transforms import train_transforms, val_transforms, collate_fn

logger = create_logger()
logger.info('Training notebook started')

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


# Set proportion sizes for subsampling the data
sizes = [0.05, 0.1, 0.2, 0.5, 1.0]

img_data = pd.read_csv('./data/power_analysis/seed_weights.csv')
train_imgs = img_data.loc[img_data['class']=='train']
val_imgs = img_data.loc[img_data['class']=='val']

logger.info("Starting training scenarios.")

for size in sizes:
    today = str(datetime.date.today())
    writer = SummaryWriter(os.path.join(tensorboard_dir, f"{base_name}_p{size}_lr{lr}_{today}"))
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
    try:
        model_name = base_name + f"_{size}"
        print(model_name)
        model = create_model(n_classes, n_obj_det=500)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        print(os.path.exists(f"./model_chkpt/{model_name}"))
        if not os.path.exists(f"./model_chkpt/{model_name}"):
            print("the directory does not exist")
            os.mkdir(f"./model_chkpt/{model_name}")
            print(os.path.exists(f"./model_chkpt/{model_name}"))
    except Exception as e:
        logger.exception('An exception occurred.')
    
    # Create training and validation datasets and dataloaders
    train_dataset = SeedDataset(
        image_dir=train_dir, 
        annotation_path=annotation_path, 
        transforms=train_transforms(), 
        subset=list(train.global_key.unique())
        )

    val_dataset = SeedDataset(
        image_dir=val_dir, 
        annotation_path=annotation_path, 
        transforms=val_transforms(), 
        subset=list(val.global_key.unique())
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        num_workers=cores,
        collate_fn=collate_fn
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=cores,
        collate_fn=collate_fn
        )
    
    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=n_epochs,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        writer=writer,
        model_name=model_name
    )

    # Calculate metrics for each model
    model.eval()

    # Instantiate metrics
    iou_metric = IntersectionOverUnion(iou_threshold=0.5, class_metrics=True, respect_labels=True)
    map_metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True, max_detection_thresholds=[100, 250, 500])

    # Calculate IOU
    try:
        iou = _iou_metrics(model=model, device=device, data_loader=val_loader, metric=iou_metric)
        map_avgs, map_class_avg, mar_class_avg = _map_metrics(model=model, device=device, data_loader=val_loader, metric=map_metric)
        
        avg_iou = iou['iou'].item()
        split_iou = iou['iou/cl_1'].item()
        seed_iou = iou['iou/cl_2'].item()
        pod_iou = iou['iou/cl_3'].item()

        writer.add_scalar('avg_iou', avg_iou)
        writer.add_scalar('split_iou', split_iou)
        writer.add_scalar('seed_iou', seed_iou)
        writer.add_scalar('pod_iou', pod_iou)

        writer.add_scalar('map_avg', map_avgs['map'])
        writer.add_scalar('map_50', map_avgs['map_50'])
        writer.add_scalar('map_75', map_avgs['map_75'])
        writer.add_scalar('map_small', map_avgs['map_small'])
        writer.add_scalar('map_medium', map_avgs['map_medium'])
        writer.add_scalar('mar_small', map_avgs['mar_small'])
        writer.add_scalar('mar_medium', map_avgs['mar_medium'])
        
        writer.add_scalar('map_split', map_class_avg['1'])
        writer.add_scalar('map_seed', map_class_avg['2'])
        writer.add_scalar('map_pod', map_class_avg['3'])

        writer.add_scalar('mar_split', mar_class_avg['1'])
        writer.add_scalar('mar_seed', mar_class_avg['2'])
        writer.add_scalar('mar_pod', mar_class_avg['3'])
    except Exception as e:
       print(e)

    writer.flush()
    writer.close()
