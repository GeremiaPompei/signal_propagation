import torch


def select_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
