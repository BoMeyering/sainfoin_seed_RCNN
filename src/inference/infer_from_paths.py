# Python module imports
import sys
import os
import torch
import pandas as pd
import numpy as np
import cv2
from glob import glob
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.ops import nms

sys.path.append('./src')

# local script imports
from config import inference_dir
from config import device, classes
from model.model import create_model
from model.train_val import train_model, _iou_metrics, _map_metrics
from utils.dataset import SeedDataset, dir_sampler
from utils.loggers import create_logger
from utils.transforms import train_transforms, val_transforms, collate_fn
from inference.predictions import tensor_img, predict, draw_boxes

state_paths = glob('./model_chkpt/frcnn_sainfoin_1.0/*_100.pth')
state_paths.sort()
print(state_paths)

image_dir = 'data/images/all_images'
image_paths = glob(image_dir+"/*.jpg")

dir_path = '.'
detection_threshold = .5

model = create_model(n_classes=4, n_obj_det=500)
model.to(device)

for path in state_paths:
    model.load_state_dict(
        torch.load(
          path,
          map_location=device
      )
    )
    model.eval()

    res_list = []

    for path in tqdm(image_paths):
        img_name = path.split('/')[-1]

        img, orig = tensor_img(img_path=path)
        outputs = predict(img, model, device=device, nms_threshold=0.2, conf_threshold=0.3)
        counts = Counter(outputs[0]['labels'].cpu().numpy())
        res_dict = {'img_name': img_name}
        for i in range(1, 4):
            cl_name = classes.get(f'{i}')
            res_dict[cl_name] = counts[i]
        res_list.append(res_dict)

        pred_img = draw_boxes(src=orig, outputs=outputs, classes=classes)
        new_img_path = os.path.join(inference_dir, f"{img_name[:-4]}.jpg")
        cv2.imwrite(new_img_path, pred_img)

    
    res_df = pd.DataFrame.from_dict(res_list)
    res_df.to_csv('./data/power_analysis/object_counts.csv')
    print(res_df)
    


            
        
