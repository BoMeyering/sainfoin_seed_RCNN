import torch

def create_optim(model=None, lr=0.1, momentum=0.8):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def create_scheduler(optimizer, gamma=0.9):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
 