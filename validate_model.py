"""
validate_model.py
Run the model against the validation set and calculate metrics
BoMeyering 2025
"""

import torch
import torchvision
import os
import cv2

from glob import glob
from config.orig_config