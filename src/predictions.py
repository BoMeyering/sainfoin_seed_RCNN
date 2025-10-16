"""
src/predictions.py
Prediction functions
BoMeyering 2025
"""

import torch
from typing import List

from ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion
from ensemble_boxes.ensemble_boxes_nms import soft_nms, nms


def process_predictions(output_list: List[dict], img_shapes: List[torch.tensor], method: str='soft_nms', **kwargs) -> List[dict]:
    """ Process a list of output dictionaries """

    if method not in ['wbf', 'soft_nms']:
        raise ValueError(f"argument 'method' must be one of 'wbf' or 'soft_nms'.")

    processed = []
    for out_dict, shape in zip(output_list, img_shapes):

        out_dict['boxes'] = _normalize_boxes(out_dict['boxes'], shape) # Normalize bbox coordinates

        # Run processing method
        if method == 'soft_nms':
            boxes, scores, labels = _run_soft_nms(input=out_dict, **kwargs)
        else:
            boxes, scores, labels = _run_wbf(input=out_dict, **kwargs)

        out_dict = {
            'boxes': _scale_boxes(torch.tensor(boxes), shape),
            'scores': torch.tensor(scores),
            'labels': torch.tensor(labels, dtype=torch.int)
        }


        processed.append(out_dict)

    return processed

def _run_wbf(input, **kwargs):
    """ Run WBF on the raw output dictionary """

    # Wrap each in a list
    for k, v in input.items():
        input[k] = [v.tolist()]

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list=input['boxes'], 
        scores_list=input['scores'], 
        labels_list=input['labels'],
        **kwargs
    )
    
    return boxes, scores, labels


def _run_soft_nms(input, **kwargs):
    """ Run WBF on the raw output dictionary """

    # Wrap each in a list
    for k, v in input.items():
        input[k] = [v.tolist()]

    boxes, scores, labels = soft_nms(
        boxes=input['boxes'], 
        scores=input['scores'], 
        labels=input['labels'],
        **kwargs
    )

    return boxes, scores, labels


def _normalize_boxes(boxes: torch.tensor, img_shape: torch.tensor):

    expanded_shape = torch.flip(img_shape, [0]).repeat(2).to(boxes.device)
    boxes = boxes / expanded_shape

    return boxes

def _scale_boxes(boxes: torch.tensor, img_shape: torch.tensor):
    expanded_shape = torch.flip(img_shape, [0]).repeat(2).to(boxes.device)
    boxes = boxes * expanded_shape

    return boxes