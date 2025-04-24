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
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from src.trainer import SupervisedTrainer
from src.loggers import setup_loggers, TBLogger
# from src.utils.dataset import SeedDataset
from src.datasets import SeedDataset

from src.model import create_model
from src.transforms import collate_fn, get_train_transforms, get_val_transforms

# CONFIG = 'config/basic_train_config.yaml'
CONFIG = 'config/tb_config.yaml'
conf = OmegaConf.load(CONFIG)
conf.base_run_name = conf.run_name

def main():
    for subset_size in conf.general.subset_size:
        
        now = datetime.now().isoformat(timespec='seconds', sep='_').replace(":", ".")
        conf.run_name = "_".join([conf.base_run_name, str(subset_size), now])
        # Setup loggers
        setup_loggers(conf)
        logger = logging.getLogger()
        logger.info(f"Read in configuration file at {CONFIG}")
        logger.info(f"Training logs are being written to {conf.directories.log_dir}")

        tb_writer = SummaryWriter(log_dir=Path(conf.directories.tensorboard_dir) / conf.run_name)
        tb_logger = TBLogger(tb_writer)

        # Initiate the model
        model = create_model(**conf.model).to(conf.device)
        logger.info(f"Create FasterRCNN model with {conf.model}")
        
        # Create optimizer and LR scheduler
        optimizer = torch.optim.Adam(model.parameters(), **conf.optimizer)
        # optimizer = torch.optim.SGD(model.parameters(), **conf.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, **conf.scheduler)
        logger.info(f"Create optimizer SGD with parameters {conf.optimizer}")
        logger.info(f"Create LR scheduler with parameters {conf.scheduler}")

        # Transforms
        train_transforms = get_train_transforms()
        val_transforms = get_val_transforms()
        logger.info(f"Create training and validation augmentation pipelines")

        # Initiate Datasets
        train_ds = SeedDataset(
            image_dir=conf.directories.train_dir, 
            label_dir=conf.directories.label_dir, 
            transforms=train_transforms, 
            subset_size=subset_size)
        
        val_ds = SeedDataset(
            image_dir=conf.directories.val_dir, 
            label_dir=conf.directories.label_dir, 
            transforms=val_transforms, 
            subset_size=subset_size)
        
        logger.info(f"Instantiated training and validation datasets.")

        # Initiate DataLoaders
        train_loader = DataLoader(dataset=train_ds, collate_fn=collate_fn, **conf.dataloader)
        val_loader = DataLoader(dataset=val_ds, collate_fn=collate_fn, **conf.dataloader)
        logger.info(f"Dataloaders created with arguments {conf.dataloader}.")

        # Create the trainer instance and train
        trainer = SupervisedTrainer(
            conf=conf, 
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            logger=logger,
            tb_logger=tb_logger)
        
        logger.info(f"Training {conf.run_name} over {subset_size * 100}% of the training and validation data.")
        trainer.train()
        logger.info(f"Training complete!")

    # Cleanup
    tb_writer.flush()
    tb_writer.close()

if __name__ == '__main__':
    main()