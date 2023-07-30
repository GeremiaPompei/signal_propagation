import torch


def select_device() -> str:
    """
    Function able to select best accelerator available.

    returns: String representation of selected device.
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
