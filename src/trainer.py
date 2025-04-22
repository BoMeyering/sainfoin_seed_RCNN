"""
src/trainer.py
Main trainer class
BoMeyering
"""

import torch
import sys
import time
import torch.nn as nn
import numpy as np
import torch.utils.tensorboard
import omegaconf
import logging

from omegaconf import OmegaConf
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
from typing import Tuple
from config.orig_config import device, n_epochs
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from src.model import create_model
from src.eval import AverageMeterSet
from src.callbacks import ModelCheckpoint

class Trainer(ABC):
	@abstractmethod
	def __init__(self, name: str):
		self.name = name
		self.meters = AverageMeterSet()
	
	@abstractmethod
	def _train_step(self):
		pass
  
	@abstractmethod
	def _train_epoch(self):
		pass
  
	@abstractmethod
	def _val_step(self):
		pass
  
	@abstractmethod
	def _val_epoch(self):
		pass
  
	@abstractmethod
	def train(self):
		pass
  

class SupervisedTrainer(Trainer):
	def __init__(
			self,
			conf: omegaconf.OmegaConf,
			model: torch.nn.Module,
			train_loader: torch.utils.data.DataLoader,
			val_loader: torch.utils.data.DataLoader,
			optimizer: torch.optim.Optimizer,
			scheduler=None,
			tb_logger: torch.utils.tensorboard.SummaryWriter=None,
			logger: logging.Logger=None
	):
		super().__init__(name=conf.run_name)
		self.conf = conf
		self.epochs = self.conf.general.epochs
		self.trainer_id = "_".join(["supervised_trainer", str(uuid4())])
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.tb_logger = tb_logger
		self.logger = logger
		self.device = self.conf.device

		# Set up train metrics
		# self.train_metrics = MetricLogger(
		# 	num_classes=self.conf.model.num_classes,
		# 	device=self.conf.device
		# )

		self.checkpoint_path = Path(conf.directories.checkpoint_dir) / self.conf.run_name
		self.checkpoint = ModelCheckpoint(filepath=self.checkpoint_path, metadata=vars(self.conf), monitor='val_loss')

		   
	def _train_step(self, batch: Tuple):
		""" Train one batch in the epoch """

		images, targets = batch
		images.to(self.device)
		targets = [
			{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets
		]

		loss_dict = self.model(images, targets)
		loss = torch.zeros(1, device=self.device)
		for loss_v in loss_dict.values():
			loss += loss_v
		
		self.meters.update("train_loss", loss.item())
	
		return loss

	def _train_epoch(self, epoch: int):

		# Reset meters and model
		self.meters.reset()
		self.model.train()

		p_bar = tqdm(range(len(self.train_loader)))

		for batch_idx, batch in enumerate(self.train_loader):
			self.optimizer.zero_grad()
			loss = self._train_step(batch)
			loss.backward()
			self.optimizer.step()

			p_bar.set_description(
				"Train Epoch: {epoch}/{epochs:3}. Iter: {batch:4}/{iterations:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
					epoch=epoch,
					epochs=self.epochs,	
					batch=batch_idx + 1,
					iterations=len(self.train_loader),
					lr=self.scheduler.get_last_lr()[0],
					loss=loss.item()
				)
			)
			p_bar.update()
		
		if self.scheduler is not None:
			self.scheduler.step()
		
		avg_loss = self.meters['train_loss'].avg

		return avg_loss
			
	@torch.no_grad()
	def _val_step(self, batch: Tuple):
		""" Validate one batch in the epoch """
		images, target = batch
		images.to(self.device)
		targets = [
			{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets
		]

		loss_dict = self.model(images)
		loss = torch.zeros(1, device=self.device)
		for loss_v in loss_dict.values():
			loss += loss_v
		
		self.meters.update("val_loss", loss.item())
		# self.metrics.update(preds, targets)

		return loss
	
	@torch.no_grad()
	def _val_epoch(self, epoch: int):

		# Reset meters and model
		self.meters.reset()
		self.val_metrics.reset()
		self.model.eval()

		p_bar = tqdm(range(len(self.val_loader)))

		for batch_idx, batch in enumerate(self.val_loader):
			loss = self._val_step(batch)

			p_bar.set_description(
				"Train Epoch: {epoch}/{epochs:3}. Iter: {batch:4}/{iterations:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
					epoch=epoch,
					epochs=self.epochs,
					batch=batch_idx + 1,
					iterations=len(self.train_loader),
					lr=self.scheduler.get_last_lr()[0],
					loss=loss.item()
				)
			)
			p_bar.update()
		
		avg_loss = self.meters['val_loss'].avg
		metrics = self.metrics.compute()

		return avg_loss
	
	def train(self):
		# Main training loop
		for epoch in range(1, self.epochs+1):
			self.logger.info("TRAINING EPOCH {epoch}")
			epoch_train_loss = self._train_epoch(epoch)

			self.logger.info("VALIDATING EPOCH {epoch}")
			epoch_val_loss = self._val_epoch(epoch)
			
			# Set the model logs
			model_logs = {
				"epoch": epoch,
				"train_loss": epoch_train_loss,
				"val_loss": epoch_val_loss,
				"model_state_dict": self.model.state_dict()
			}

			self.logger.info(f"Epoch: {epoch}. Train Loss: {epoch_train_loss}. Val Loss: {epoch_val_loss}")

			# Checkpoint
			self.checkpoint(epoch=epoch, logs=model_logs)

