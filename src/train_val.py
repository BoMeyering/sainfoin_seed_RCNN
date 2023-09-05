import torch
import logging
from tqdm import tqdm
import numpy as np
import time

from config import device, n_epochs


def _train(model, optimizer, device, data_loader):
  
  prog_bar = tqdm(data_loader, total=len(data_loader))
  loss_list = []

  for i, data in enumerate(prog_bar):
    optimizer.zero_grad()
    images, targets = data

    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]

    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())

    loss_value = losses.item()
    loss_list.append(loss_value)
    losses.backward()
    optimizer.step()
    prog_bar.set_description(desc=f"Train Loss: {loss_value: .4f}")

  return loss_list


def _validate(model, device, data_loader):
  
  prog_bar = tqdm(data_loader, total=len(data_loader))
  loss_list = []

  for i, data in enumerate(prog_bar):
    images, targets = data
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]

    with torch.no_grad():
      loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()
    loss_list.append(loss_value)
    prog_bar.set_description(desc=f"Val Loss: {loss_value:.4f}")

  return loss_list


def train_model(model, optimizer, scheduler, n_epochs, device, train_loader, val_loader, logger, writer, model_name):
  
  # send model to the device
  model.to(device)
  train_loss_list = []
  val_loss_list = []
  lr_list = []
  logger.info(f"Initializing training sequence")
  for epoch in range(1, n_epochs+1):
    logger.info(f"Epoch: {epoch}\nLR: {scheduler.get_last_lr()}")
    lr_list.append(scheduler.get_last_lr())
    train_losses = _train(model=model, optimizer=optimizer, device=device, data_loader=train_loader)
    val_losses = _validate(model=model, device=device, data_loader=val_loader)
    avg_train_loss = np.mean(np.array(train_losses))
    avg_val_loss = np.mean(np.array(val_losses))
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    writer.add_scalar('train_loss', avg_train_loss, epoch)
    writer.add_scalar('val_loss', avg_val_loss, epoch)
    scheduler.step()
    time.sleep(5)
    if epoch==(n_epochs):
      torch.save(model.state_dict(), f"./model_chkpt/model{epoch}_{model_name}.pth")

  return train_loss_list, val_loss_list, lr_list