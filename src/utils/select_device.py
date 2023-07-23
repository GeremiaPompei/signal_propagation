import torch


def select_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
