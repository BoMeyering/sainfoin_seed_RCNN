import os
import torch
import argparse
import logging

from omegaconf import OmegaConf
from torchinfo import summary
from torchviz import make_dot
from torchview import draw_graph


from src.model import create_model
from src.transforms import get_val_transforms

# CONFIG = 'config/basic_train_config.yaml'
CONFIG = 'config/basic_train_config.yaml'
conf = OmegaConf.load(CONFIG)
conf.device = 'cpu'
conf.base_run_name = conf.run_name

model = create_model(**conf.model).to(conf.device)
model.eval()

print(model)

summary(model, input_size=(1, 3, 1024, 1024))

# x = torch.randn(1, 3, 1024, 1024)
# y = model(x)
# make_dot(y, params=dict(model.named_parameters())).render("model_graph", format="png")

# graph = draw_graph(model, input_size=(1, 3, 1024,1024))
# graph.visual_graph.render("model_architecture", format="png")

# from src.datasets import SeedDataset

# val_transforms = get_val_transforms()

# train_ds = SeedDataset(
#     image_dir=conf.directories.train_dir, 
#     label_dir=conf.directories.label_dir, 
#     transforms=val_transforms, 
#     subset_size=1.0
# )

# w_means = []
# h_means = []

# for idx in range(len(train_ds)):
#     print(idx)

#     _, targets = train_ds[idx]

#     boxes = targets['boxes']
#     widths = boxes[:,2] - boxes[:,0]
#     heights = boxes[:,3] - boxes[:,1]

    

#     w_means.append(widths.mean())
#     h_means.append(heights.mean())

# print(
# torch.mean(torch.tensor(w_means)),
# torch.mean(torch.tensor(h_means))
# )