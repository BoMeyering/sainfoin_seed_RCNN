import torch
import numpy as np
import cv2
import torchvision
from torchvision.ops import nms
from typing import Optional

def tensor_img(img_path: str) -> tuple[torch.Tensor, np.array]:
    """
    Loads an image path and returns a scaled tensor
    """
    orig = cv2.imread(img_path)
    if type(orig) == None:
        raise TypeError(f"Can't read/open the path {img_path}. Check the path integrity")
    img = orig.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = torch.tensor(img, dtype=torch.float)
    img = torch.unsqueeze(img, 0)
    
    return img, orig

@torch.no_grad()
def predict(img: torch.Tensor, model: torchvision.models.detection.faster_rcnn.FasterRCNN, device: str='cpu', nms_threshold: Optional[float]=None, conf_threshold: Optional[float]=None):
    """
    Predicts the bounding boxes for an image and applies non-maximum supression
    Parameters:
        img:
        model:
        device:
        nms_threshold:
        conf_threshold:
    
    Returns:
        
    """

    img_device = img.get_device()
    if img_device != device:
        img = img.to(device)
    
    outputs = model(img)

    if conf_threshold is not None:
        score_idx = torch.where(outputs[0]['scores'] >= conf_threshold)
        outputs[0]['boxes'] = outputs[0]['boxes'][score_idx]
        outputs[0]['labels'] = outputs[0]['labels'][score_idx]
        outputs[0]['scores'] = outputs[0]['scores'][score_idx]
    
    if nms_threshold is not None:
        nms_idx = nms(boxes=outputs[0]['boxes'], scores=outputs[0]['scores'], iou_threshold=nms_threshold)
        outputs[0]['boxes'] = outputs[0]['boxes'][nms_idx]
        outputs[0]['labels'] = outputs[0]['labels'][nms_idx]
        outputs[0]['scores'] = outputs[0]['scores'][nms_idx]

    return outputs


def draw_boxes(src: np.ndarray, outputs: list, classes: dict) -> np.ndarray:
    """
    
    """
    boxes = outputs[0]['boxes'].data.cpu().numpy().astype(np.int32)
    scores = outputs[0]['scores'].data.cpu().numpy()
    labels = outputs[0]['labels'].data.cpu().numpy()
    draw_boxes = boxes.copy()

    # get all the predicited class names
    pred_classes = [classes[f'{i}'] for i in labels]

    # draw predictions, classes and scores
    for j, box in enumerate(draw_boxes):
        cv2.rectangle(src,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (255, 255, 0),
                      1)
        cv2.putText(src, 
                    pred_classes[j] + f" {np.round(scores[j], 4): .4f}",
                    (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 150, 255),
                    1, 
                    lineType=cv2.LINE_AA)

    return src

def show_img(img):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.destroyAllWindows()