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
  val_iter = iter(data_loader)
  while True:
    try:
      images, targets = next(val_iter)
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]
      preds = model(images)
      with torch.no_grad():
        metric.update(preds=preds, target=targets)
        # m_output = m.compute()
        # print(m_output)
        # m.reset()
    except StopIteration:
      break

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

# def _val_metrics2(model, device, data_loader, metrics: list):
#   val_iter = iter(data_loader)
#   while True:
#     try:
#       for m in metrics:
#       images, targets = next(val_iter)
#       images = list(image.to(device) for image in images)
#       targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]
#       preds = model(images)
#       with torch.no_grad():
#         for m in metrics:
#           m.update(preds=preds, target=targets)
#           m_output = m.compute()
#           print(m_output)
#           m.reset()
          
#     except StopIteration:
#       break
  


def train_model(model, optimizer, scheduler, n_epochs, device, train_loader, val_loader, logger, writer, model_name):
  
  # send model to the device
  model.to(device)
  train_loss_list = []
  val_loss_list = []
  lr_list = []
  logger.info(f"Initializing training sequence")
  iou_metric = IntersectionOverUnion(iou_threshold=0.5, class_metrics=True, respect_labels=True)
  map_metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True, max_detection_thresholds=[100, 250, 500])
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
    writer.add_scalar('lr', scheduler.get_last_lr()[0])
    print("Epoch results")
    print(f"Avg train_loss: {avg_train_loss}")
    print(f"Avg val_loss: {avg_val_loss}")
    scheduler.step()
    time.sleep(5)
    if epoch % 2 == 0:
      torch.save(model.state_dict(), f"./model_chkpt/{model_name}/{model_name}_{epoch}.pth")
    
    # metric calc on val set
    # model.eval()
    # _val_metrics(model=model, device=device, data_loader=train_loader, metrics=[map_metric])
    # _iou_fucntional(model=model, device=device, data_loader=val_loader)
    # total_iou = iou_metric.compute()
    # total_map = map_metric.compute()
    # print(total_iou)
    # print(total_map)
    # iou_metric.reset()
    # map_metric.reset()
    # model.train()
      

  return train_loss_list, val_loss_list, lr_list