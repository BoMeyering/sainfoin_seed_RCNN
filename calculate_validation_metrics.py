"""
src/infer.py
Inference with bbox processing
BoMeyering 2025
"""

import sys
import logging
import json
import tqdm
import torch
import pandas as pd
import numpy as np
import cv2
import datetime
from pprint import pprint
from glob import glob
from collections import Counter
from tqdm import tqdm
from pathlib import Path
from torchvision.ops import nms
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from torchmetrics import MetricCollection

# local script imports
from src.datasets import SeedDataset
from src.inference.predictions import draw_boxes, show_img
from src.transforms import get_inf_transforms
from src.model import create_model
from src.predictions import process_predictions


# Set constants
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
IMG_DIR = 'data/images/val'
CHECKPOINTS = {
    'subset_0.05': 'model_checkpoints/frcnn_sainfoin_0.05/frcnn_sainfoin_0.05_100.pth',
    'subset_0.1': 'model_checkpoints/frcnn_sainfoin_0.1/frcnn_sainfoin_0.1_100.pth',
    'subset_0.2':'model_checkpoints/frcnn_sainfoin_0.2/frcnn_sainfoin_0.2_100.pth',
    'subset_0.5': 'model_checkpoints/frcnn_sainfoin_0.5/frcnn_sainfoin_0.5_100.pth',
    'subset_1.0':'model_checkpoints/frcnn_sainfoin_1.0/frcnn_sainfoin_1.0_100.pth'
}
CLASSES = {
    '0': 'background',
    '1': 'split',
    '2': 'seed',
    '3': 'pod'
}

# Confidence Threshold - initial low confidence filtering
CONF_THRESHOLD = 0.1

# Set IOU Threshold for class agnostic filtering NMS
NMS_THRESHOLD = 0.5

# Bbox Processing keyword arguments
# Weighted Box Fusion kwargs
proc_kwargs = {
    'iou_thr': 0.3,
    # 'thresh': 0.1,
    'skip_box_thr': 0.1
}

# Setup stdout logger
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(sys.stdout,)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)
root_logger.setLevel(logging.INFO)

# Instantiate model
model = create_model(4, 500)
# model = model.to(DEVICE)

@torch.no_grad()
def main():
    """ Run Prediction and Evaluation Metrics """

    root_logger.info("Starting checkpoint evaluation")

    # Set up standard inference transforms
    transforms = get_inf_transforms()

    # Seed dataset
    dataset = SeedDataset(
        image_dir=IMG_DIR, 
        label_dir='data/annotations/coco_annotations.json', 
        transforms=transforms
    )

    # Create MetricCollection
    metrics = MetricCollection(
        IntersectionOverUnion(iou_threshold=0.5, class_metrics=True),
        MeanAveragePrecision(iou_type='bbox', max_detection_thresholds=[100, 300, 500], class_metrics=True, average='macro')
    ).to(DEVICE)

    # Run through each checkpoint
    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
        root_logger.info(f"Loading checkpoint {checkpoint_name}")
        # Instantiate model
        model = create_model(4, 500)
        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location='cuda',
                weights_only=False
            )
        )
        model = model.to(DEVICE)
        model.eval()

        # Collect inference times
        inference_times = []

        # Iterate through dataset and run inference
        p_bar = tqdm(range(len(dataset)))

        for idx in range(len(dataset)):
            img, targets = dataset[idx]

            p_bar.set_description(
				"Checkpoint: {checkpoint_name}. Img ID: {img_id}".format(
					checkpoint_name=checkpoint_name,
                    img_id=targets['img_id']
				)
			)
            p_bar.update()

            # Send data to device
            img = img.to(DEVICE)
            targets = [
                {k: (v if k == 'img_id' else v.to(DEVICE)) for k, v in targets.items()}
            ]

            img_shape = [torch.tensor(img.shape[1:])]
            if checkpoint_name == 'subset_1.0':
                cv_img = np.moveaxis(img.cpu().numpy(), source=0, destination=2)
                cv_img = (cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
            
            # Add batch dimension and send to model
            img = img.unsqueeze(0)

            t0 = datetime.datetime.now()
            output = model(img)
            t1 = datetime.datetime.now()
            inference_times.append((t1 - t0))

            # Filter out low confidence bboxes
            conf_idx = torch.where(output[0]['scores'] > CONF_THRESHOLD)
            output = [{k: v[conf_idx] for k,v in output[0].items()}]

            # Run class agnostic NMS
            keep_idx = nms(output[0]['boxes'], scores=output[0]['scores'], iou_threshold=NMS_THRESHOLD)
            output = [{k: v[keep_idx] for k,v in output[0].items()}]

            output = process_predictions(output_list=output, img_shapes=img_shape, method='wbf', **proc_kwargs)
            output = [
                {k: v.to(DEVICE) for k, v in o.items()} for o in output
            ]

            # Update metrics
            metrics.update(preds=output, target=targets)

            if checkpoint_name == 'subset_1.0':
                # Draw bboxes
                wbf_img = draw_boxes(src=cv_img.copy(), outputs=output, classes=CLASSES)

                # Write out the image
                img_id = targets[0]['img_id']
                out_path = Path('outputs/val') / (img_id + ".jpg")
                cv2.imwrite(filename=out_path, img=wbf_img)

        metric_output = metrics.compute()
        metric_output = {k: v.cpu().numpy().tolist() for k, v in metric_output.items()}
        root_logger.info(pprint(metric_output))

        inference_times = [diff.seconds + diff.microseconds/1000000 for diff in inference_times]

        root_logger.info(f"Mean Inference Time: {np.array(inference_times).mean()} seconds")

        metric_output['mean_inference_time'] = np.array(inference_times).mean()

        with open(f'outputs/{checkpoint_name}_validation_metrics.json', 'w') as f:
            json.dump(metric_output, f)

        metrics.reset()


if __name__ == '__main__':
    main()

# with torch.no_grad():
#     for idx in range(len(dataset)):
#         img, targets = dataset[idx]
#         print(idx, targets['img_id'])
#         # Send data to device
#         img = img.to(DEVICE)
#         targets = [
# 			{k: (v if k == 'img_id' else v.to(DEVICE)) for k, v in targets.items()}
# 		]

#         cv_img = np.moveaxis(img.cpu().numpy(), source=0, destination=2)
#         cv_img = (cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
#         img_shape = [torch.tensor(cv_img.shape[:2])]

#         # Add batch dimension and send to model
#         img = img.unsqueeze(0)

#         t0 = datetime.datetime.now()
#         output = model(img)
#         t1 = datetime.datetime.now()
#         inference_times.append((t1 - t0))

#         # Move targets and outputs to cpu
#         targets = [
# 			{k: (v if k == 'img_id' else v.cpu()) for k, v in t.items()} for t in targets
# 		]

#         output = [
#             {k: v.cpu() for k, v in output[0].items()}
#         ]

#         # Filter out low confidence bboxes
#         conf_idx = torch.where(output[0]['scores'] > CONF_THRESHOLD)
#         output = [{k: v[conf_idx] for k,v in output[0].items()}]

#         # Run class agnostic NMS
#         keep_idx = nms(output[0]['boxes'], scores=output[0]['scores'], iou_threshold=NMS_THRESHOLD)
#         output = [{k: v[keep_idx] for k,v in output[0].items()}]

#         # Weighted Box Fusion kwargs
#         proc_kwargs = {
#             'iou_thr': 0.3,
#             # 'thresh': 0.1,
#             'skip_box_thr': 0.1
#         }

#         output = process_predictions(output_list=output, img_shapes=img_shape, method='wbf', **proc_kwargs)

#         # Update metrics
#         metrics.update(preds=output, target=targets)

#         # Draw bboxes
#         wbf_img = draw_boxes(src=cv_img.copy(), outputs=output, classes=CLASSES)

#         # Write out the image
#         img_id = targets[0]['img_id']
#         out_path = Path('outputs/test') / (img_id + ".jpg")
#         cv2.imwrite(filename=out_path, img=wbf_img)
# metric_output = metrics.compute()
# metric_output = {k: v.numpy().tolist() for k, v in metric_output.items()}
# pprint(metric_output)

# with open('outputs/validation_metrics.json', 'w') as f:
#     json.dump(metric_output, f)

# inference_times = [diff.seconds + diff.microseconds/1000000 for diff in inference_times]

# print(f"Mean Inference Time: {np.array(inference_times).mean()}")

