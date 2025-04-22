import torch

train_dir = './data/images/train'
val_dir = './data/images/val'
test_dir = './data/images/test'
annotation_path = './data/annotations/coco_annotations.json'
chkpt_dir = './model_chkpt'
tensorboard_dir = './runs'
log_dir = './logs'
inference_dir = './inference'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cores = 4
classes = {
    '0': 'background',
    '1': 'split',
    '2': 'seed',
    '3': 'pod'
}
n_classes = 4
resize_to = 3000
n_epochs = 100
batch_size = 1
base_name = 'frcnn_sainfoin'
lr = 0.05
momentum = 0.9
gamma = 0.9
