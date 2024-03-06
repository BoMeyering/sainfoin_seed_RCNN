# metrics.py
# Defines the IOU functions needed for output

import torch
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.functional.detection.iou import intersection_over_union
from scipy.optimize import linear_sum_assignment

def fast_iou(boxA, boxB):
  """
  IOU function adapted from https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
  Bbox format = (xmin, ymin, xmax, ymax)
  Parameters:
    boxA: The ground truth bounding box
    boxB: The predicted bounding box
  
  Returns:
    An IOU value in [0, 1]
  """
    
  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou

def match_boxes(preds: torch.Tensor, targets: torch.Tensor, iou_thresh: float=0.0, on_device: bool=True, return_misses:bool=False):
    """
    match_boxes function adapted from https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
    Parameters:
      preds: torch.Tensor of shape (N, 4) in format (xmin, ymin, xmax, ymax)
      targets: torch.Tensor of shape (N, 4) in format (xmin, ymin, xmax, ymax)
      iou_thresh: iou threshold to determine pred/target bbox matches
      return_misses: bool type, return FP and FN indices?

    Returns:
      A tuple of two torch.Tensors of shape ()
    """
    if on_device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_pred = preds.shape[0]
    n_target = targets.shape[0]
    
    max_size = torch.max(torch.tensor([n_pred, n_target]))

    iou_matrix = torch.zeros(size=[max_size, max_size],
                             dtype=torch.float32)
    for i in range(n_target):
       for j in range(n_pred):
          iou = fast_iou(targets[i,:], preds[j,:])
          iou_matrix[i,j] = iou
          
    idx_true, idx_pred = linear_sum_assignment(1 - iou_matrix)
    
    ious = iou_matrix[idx_true, idx_pred]
    
    sel_pred = idx_pred < n_pred
    idx_pred_actual = idx_pred[sel_pred]
    idx_gt_actual = idx_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > iou_thresh)

    label = sel_valid.numpy().astype(int)
    
    return torch.tensor(idx_gt_actual[sel_valid]), torch.tensor(idx_pred_actual[sel_valid]), ious_actual[sel_valid], label

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