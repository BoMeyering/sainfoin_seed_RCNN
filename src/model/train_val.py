import torch
import logging
from tqdm import tqdm
import numpy as np
import time
import torch.nn as nn
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from torchmetrics.functional.detection import intersection_over_union
import sys
import scipy

sys.path.append('./src')

from config import device, n_epochs

def _train(model, optimizer, device, data_loader):
  
  prog_bar = tqdm(data_loader, total=len(data_loader))
  loss_list = []
  ind_loss_list = []

  for i, data in enumerate(prog_bar):
    optimizer.zero_grad()
    images, targets = data

    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]

    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())

    loss_value = losses.item()
    loss_list.append(loss_value)
    ind_loss_list.append(loss_dict)
    losses.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 4.0)
    optimizer.step()
    prog_bar.set_description(desc=f"Train Loss: {loss_value: .4f}")

  return loss_list


def _validate(model, device, data_loader):
  
  prog_bar = tqdm(data_loader, total=len(data_loader))
  loss_list = []
  ind_loss_list = []

  for i, data in enumerate(prog_bar):
    images, targets = data
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]

    with torch.no_grad():
      loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()
    loss_list.append(loss_value)
    ind_loss_list.append(loss_dict)
    prog_bar.set_description(desc=f"Val Loss: {loss_value:.4f}")

  return loss_list

def _iou_metrics(model, device, data_loader, metric):
  batches = iter(data_loader)
  while True:
    try:
      images, targets = next(batches)
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]
      preds = model(images)
      with torch.no_grad():
        metric.update(preds=preds, target=targets)
    except StopIteration:
      break
  
  iou = metric.compute()
  metric.reset()

  return iou

def _map_metrics(model, device, data_loader, metric):
  batches = iter(data_loader)
  map_res = []
  while True:
    try:
      images, targets = next(batches)
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]
      preds = model(images)
      with torch.no_grad():
        map_scores = metric(preds=preds, target=targets)
        map_res.append(map_scores)
        metric.reset()
    except StopIteration:
      break
  maps = {'map': [],
          'map_50': [],
          'map_75': [], 
          'map_small': [],
          'map_medium': [],
          'map_large': [], 
          'mar_small': [],
          'mar_medium': [],
          'mar_large': [],
          'map_classes': {
            '0': [],
            '1': [],
            '2': [],
            '3': []
          },
          'mar_100_classes': {
            '0': [],
            '1': [],
            '2': [],
            '3': []
          }
        }
  
  for i in map_res:
    maps['map'].append(i['map'].item())
    maps['map_50'].append(i['map_50'].item())
    maps['map_75'].append(i['map_75'].item())
    if i['map_small'].item() != -1:
      maps['map_small'].append(i['map_small'].item())
    if i['map_medium'].item() != -1:
      maps['map_medium'].append(i['map_medium'].item())
    if i['map_large'].item() != -1:
      maps['map_large'].append(i['map_large'].item())
    if i['mar_small'].item() != -1:
      maps['mar_small'].append(i['mar_small'].item())
    if i['mar_medium'].item() != -1:
      maps['mar_medium'].append(i['mar_medium'].item())
    if i['mar_large'].item() != -1:
      maps['mar_large'].append(i['mar_large'].item())
    for k, cl in enumerate(i['classes']):
      cl_map = i['map_per_class'][k].item()
      if cl_map != -1:
        maps['map_classes'][f'{cl.item()}'].append(cl_map)
      
      cl_mar = i['mar_100_per_class'][k].item()
      if cl_mar != -1:
        maps['mar_100_classes'][f'{cl.item()}'].append(cl_mar)
  map_avgs = {k: np.mean(v) for k, v in maps.items() if type(v) != dict}
  map_class_avgs = {k: np.mean(v) for k, v in maps['map_classes'].items()}
  mar_class_avgs = {k: np.mean(v) for k, v in maps['mar_100_classes'].items()}

  return map_avgs, map_class_avgs, mar_class_avgs

def _iou_fucntional(model, device, data_loader):
  val_iter = iter(data_loader)
  while True:
    try:
      images, targets = next(val_iter)
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v, in t.items() if k in ['boxes', 'labels']} for t in targets]
      # print(targets[0]['boxes'])
      preds = model(images)
      # print(preds[0]['boxes'])
      with torch.no_grad():
        iou_matrix = intersection_over_union(preds=preds[0]['boxes'], target=targets[0]['boxes'], aggregate=False).cpu().numpy()
        idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)
        print(idxs_true, idxs_pred)
        # print(iou_matrix)

                                             
    except StopIteration:
      break


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
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    print("Epoch results")
    print(f"Avg train_loss: {avg_train_loss}")
    print(f"Avg val_loss: {avg_val_loss}")
    scheduler.step()
    time.sleep(5)
    if epoch % 2 == 0:
      torch.save(model.state_dict(), f"./model_chkpt/{model_name}/{model_name}_{epoch}.pth")
      
  return train_loss_list, val_loss_list, lr_list