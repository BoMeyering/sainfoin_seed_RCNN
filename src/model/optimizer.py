import torch

def create_optim(model=None, lr=0.1, momentum=0.8):
    """
    Creates a standard SGD optimizer with momentum and regularization
    """
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0005)

def create_adam_optim(model=None, lr=0.001):
    """
    Creates an ADAM optimizer
    """
    return torch.optim.Adam(model.parameters(), lr=lr)

def create_scheduler(optimizer, gamma=0.9):
    """
    Creates an exponential learning rate scheduler
    """
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
 