import torch

def create_optim(model=None, lr=0.1, momentum=0.8):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0005)

def create_adam_optim(model=None, lr=0.001):
    return torch.optim.Adam(model.parameters(), lr=lr)

def create_scheduler(optimizer, gamma=0.9):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
 