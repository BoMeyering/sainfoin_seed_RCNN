"""
train.py
Main training script
BoMeyering 2025
"""

import os
import torch
import argparse
import logging

from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.trainer import SupervisedTrainer
from src.loggers import setup_loggers
# from src.utils.dataset import SeedDataset
from src.datasets import SeedDataset

from src.model import create_model
from src.utils.transforms import collate_fn

CONFIG = 'config/basic_train_config.yaml'

def main():
    conf = OmegaConf.load(CONFIG)
    now = datetime.now().isoformat(timespec='seconds', sep='_').replace(":", ".")
    conf.run_name = "_".join([conf.run_name, now])
    # Setup loggers
    setup_loggers(conf)
    logger = logging.getLogger()
    logger.info(f"Read in configuration file at {CONFIG}")
    logger.info(f"Training logs are being written to {conf.directories.log_dir}")

    

    # Initiate the model
    model = create_model(**conf.model)
    logger.info(f"Create FasterRCNN model with {conf.model}")
    
    # Create optimizer and LR scheduler
    optimizer = torch.optim.SGD(model.parameters(), **conf.optimizer)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, **conf.scheduler)
    logger.info(f"Create optimizer SGD with parameters {conf.optimizer}")
    logger.info(f"Create LR scheduler with parameters {conf.scheduler}")



    ds = SeedDataset(conf.directories.train_dir, label_dir=conf.directories.label_dir)


    img = ds[1]

    print(img)



    # Initiate Datasets
    # train_ds = SeedDataset(image_dir=conf.directories.train_dir, annotation_path=conf.directories.label_dir)
    # val_ds = SeedDataset(image_dir=conf.directories.val_dir, annotation_path=conf.directories.label_dir)

    # # Initiate DataLoaders
    # train_loader = DataLoader(dataset=train_ds, collate_fn=collate_fn, **conf.dataloader)
    # val_loader = DataLoader(dataset=val_ds, collate_fn=collate_fn, **conf.dataloader)

    # trainer = SupervisedTrainer(conf=conf, model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler, logger=logger)


    # trainer.train()

if __name__ == '__main__':
    main()