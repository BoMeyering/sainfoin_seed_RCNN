# infer.py
# Infer results from a set of images

import sys
# sys.path.append('./src')

import argparse
import os
import torch
import pandas as pd
import numpy as np
import cv2
from glob import glob
# from src.model.model import create_model
from collections import Counter
from tqdm import tqdm
from pathlib import Path
# from sklearn.model_selection import train_test_split
# from torchvision.ops import nms

# local script imports
# from src.loggers import inference_dir
# from src.loggers import device, classes
# from model.model import create_model
# from model.train_val import train_model
# from model.metrics import _iou_metrics, _map_metrics
# from utils.dataset import SeedDataset, dir_sampler
# from utils.loggers import create_logger
# from utils.transforms import train_transforms, val_transforms, collate_fn
from src.inference.predictions import draw_boxes, show_img
from src.transforms import get_inf_transforms
from src.model import create_model

# parser = argparse.ArgumentParser(description='Make inference on some images')
# parser.add_argument('img_dir', type=str, help='The relative path to the image directory')
# parser.add_argument('chkpt', type=str, help='The relative path to the model state dict')
# parser.add_argument('-t', '--threshold', dest='nms', default=0.5, type=float, help='The threshold to use for non-maximum supression')
# parser.add_argument('-c', '--confidence', dest='conf', default=0.5, type=float, help='The minimum confidene in the predictions to be returned')

# args = parser.parse_args()

# img_dir = args.img_dir
# imgs = glob("*.jpg", root_dir=img_dir)


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = create_model(n_classes=4, n_obj_det=500)
# model.to(device)

# model.load_state_dict(
#         torch.load(
#           args.chkpt,
#           map_location=device
#       )
#     )

# model.eval()

# res_list = []
# for name in tqdm(imgs):
#     path = os.path.join(img_dir, name)
#     print(path)
#     img, orig = tensor_img(img_path=path)
#     outputs = predict(img, model, device=device, nms_threshold=args.nms, conf_threshold=args.conf)
#     counts = Counter(outputs[0]['labels'].cpu().numpy())
#     res_dict = {'img_name': name}
#     for i in range(1, 4):
#         cl_name = classes.get(f'{i}')
#         res_dict[cl_name] = counts[i]
#     res_list.append(res_dict)

#     pred_img = draw_boxes(src=orig, outputs=outputs, classes=classes)
#     new_img_path = os.path.join(inference_dir, f"{name[:-4]}.jpg")
#     cv2.imwrite(new_img_path, pred_img)

    
# res_df = pd.DataFrame.from_dict(res_list)
# res_df.to_csv('./inference/inference_object_counts.csv')

# print("Image inference complete!")

TEST_DIR = 'data/images/test'
CHECKPOINT = 'model_checkpoints/basic_fasterrcnn_0.2_2025-04-23_19.59.35_epoch_79_vloss-0.373212.pth'
# CHECKPOINT = 'model_checkpoints/frcnn_sainfoin_1.0/frcnn_sainfoin_1.0_100.pth'
CLASSES = {
    '0': 'bakcground',
    '1': 'split',
    '2': 'seed',
    '3': 'pod'
}

model = create_model(4, 500)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location='cpu',
        weights_only=False
    )['model_state_dict']
)
model.eval()

transforms = get_inf_transforms()

img_names = [name for name in glob('*', root_dir=TEST_DIR) if name.lower().endswith(('jpg', 'jpeg'))]
with torch.no_grad():
    for img_name in img_names:
        path = Path(TEST_DIR) / img_name
        img = cv2.imread(path)
        img_copy = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = [transforms(image=img)['image']]

        output = model(img)[0]
        
        idx = output['scores'] >= 0.33333

        for k, v in output.items():
            output[k] = output[k][idx]
        
        labeled_img = draw_boxes(src=img_copy, outputs=[output], classes=CLASSES)

        # show_img(labeled_img)
        out_path = Path('outputs/test') / img_name

        cv2.imwrite(out_path, labeled_img)



        


    

