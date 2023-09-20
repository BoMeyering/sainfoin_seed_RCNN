import torch
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.functional.detection import intersection_over_union
from scipy.optimize import linear_sum_assignment

def fast_iou(boxA, boxB):
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.
    
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
    preds: torch.Tensor of shape (N, 4) in format (xmin, ymin, xmax, ymax)
    targets: torch.Tensor of shape (N, 4) in format (xmin, ymin, xmax, ymax)
    iou_thresh: iou threshold to determine pred/target bbox matches
    return_misses: bool type, return FP and FN indices?

    return: a tuple of two torch.Tensors of shape ()
    
    """
    if on_device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_pred = preds.shape[0]
    n_target = targets.shape[0]
    
    min_iou = 0.0

    max_size = torch.max(torch.tensor([n_pred, n_target]))

    iou_matrix = torch.zeros(size=[max_size, max_size],
                             dtype=torch.float32)
    for i in range(n_target):
       for j in range(n_pred):
        #   metric = IntersectionOverUnion(iou_threshold=iou_thresh).to(device)
          iou = fast_iou(targets[i,:], preds[j,:])
          iou_matrix[i,j] = iou
          
    idx_true, idx_pred = linear_sum_assignment(1 - iou_matrix)
    # print(idx_true, idx_pred)
    
    ious = iou_matrix[idx_true, idx_pred]
    # print(ious)
    
    
    sel_pred = idx_pred < n_pred
    idx_pred_actual = idx_pred[sel_pred]
    idx_gt_actual = idx_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > iou_thresh)
    # print(sel_valid)
    label = sel_valid.numpy().astype(int)
    
    return torch.tensor(idx_gt_actual[sel_valid]), torch.tensor(idx_pred_actual[sel_valid]), ious_actual[sel_valid], label 
    # return torch.tensor(idx_true), torch.tensor(idx_pred), ious
  
  

preds = torch.tensor([[120, 134, 234, 256],
                      [457, 234, 473, 245],
                      [10, 345, 23, 362]],
                      dtype=torch.float32,
                      device='cpu')

targets = torch.tensor([[457, 234, 473, 245],
                        [118, 132, 233, 258],
                        [15, 355, 29, 357],
                        [542, 342, 569, 375]],
                      dtype=torch.float32,
                      device='cpu')

wo_matching = intersection_over_union(preds, targets, aggregate=True, iou_threshold=0.5)
print(wo_matching)



idx_true, idx_pred, ious, label = match_boxes(preds=preds, targets=targets)

print(idx_true, idx_pred, ious, label)
preds = preds[idx_pred]

# metric = IntersectionOverUnion()
# metric.update(preds = )
