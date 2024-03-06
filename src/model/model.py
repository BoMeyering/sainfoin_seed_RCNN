# model.py
# Holds definition of FRCNN_50 model

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(n_classes: int, n_obj_det: int):
    """
    Parameters:
        n_classes (int): The number of classes to detect.
        n_obj_det (int): The maximum number of objects the model will detect in one image.

    Returns:
        A pretrained Faster RCNN with anchor sizes set up
        to detect smaller objects in the image, a new model head to detect our classes,
        and change the model head to be able to detect up to 500 objects in each image.
    """

    # Set new anchor sizes
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    # Anchor generator
    anchor_generator = AnchorGenerator(
        anchor_sizes,
        aspect_ratios
        )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights='DEFAULT'
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    # increase number of detections per image
    model.roi_heads.detections_per_img = n_obj_det
    # switch in new anchor generator object
    model.rpn.anchor_generator = anchor_generator

    return model
    