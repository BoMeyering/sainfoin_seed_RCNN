# train_val.py
# Define the train and validate loops

import torch
import sys
import time
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from config.orig_config import device, n_epochs

sys.path.append('./src')



def _train(model, optimizer, device, data_loader):
  
  prog_bar = tqdm(data_loader, total=len(data_loader))
  loss_list = []

  for i, data in enumerate(prog_bar):
    optimizer.zero_grad()
    images, targets = data

    images = [image.to(device) for image in images]

    loss_dict = model(images, targets)
    losses = torch.zeros(1, device=device)
    for loss in loss_dict.values():
      losses += loss

    # grab total loss and append to loss_list
    loss_list.append(losses.item())

    # backwards pass with gradient clipping
    losses.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 4.0)

    # update weights
    optimizer.step()

    prog_bar.set_description(desc=f"Train Loss: {losses.item(): .4f}")

  return loss_list

@torch.no_grad()
def _validate(model, device, data_loader):
  
  prog_bar = tqdm(data_loader, total=len(data_loader))
  loss_list = []

  for i, data in enumerate(prog_bar):
    images, targets = data
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]

    loss_dict = model(images, targets)
    losses = torch.zeros(1, device=device)
    for loss in loss_dict.values():
      losses += loss

    # grab total loss and append to loss_list
    loss_list.append(losses.item())

    prog_bar.set_description(desc=f"Val Loss: {losses.item():.4f}")

  return loss_list


def train_model(model, optimizer, scheduler, n_epochs, device, train_loader, val_loader, logger, writer, model_name):
  """
  Main training function

  Parameters:
    model: The Torch model to train
    optimizer: An optimizer i.e. SGD, Adam, etc.
    scheduler: The learning rate scheduler
    n_epochs [int]: The number of epochs to train the model
    device [str]: Either 'cpu' or 'cuda'
    train_loader:
    val_loader:
    logger: 
    writer:
    model_name: 

  Returns:
    train_loss_list: Average training loss value for each epoch
    val_loss_list: Average validation loss value for each epoch
    lr_list: Learning rate used for each epoch
  """
  
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
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    print("Epoch results")
    print(f"Avg train_loss: {avg_train_loss}")
    print(f"Avg val_loss: {avg_val_loss}")
    scheduler.step()
    time.sleep(5)
    if epoch % 2 == 0:
      torch.save(model.state_dict(), f"./model_chkpt/{model_name}/{model_name}_{epoch}.pth")
      
  return train_loss_list, val_loss_list, lr_list