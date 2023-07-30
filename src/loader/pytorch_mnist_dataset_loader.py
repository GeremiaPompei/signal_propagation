import os
from typing import Callable

import torch


def pytorch_mnist_dataset_loader(
        dir_name: str,
        dataset_loader_func: Callable,
        device='cpu'
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Function able to download pytorch MNIST like dataset and return it.
    :param dir_name: Name of directory where save local dataset.
    :param dataset_loader_func: Function able to download and load dataset.
    :param device: Accelerator where allocate the dataset.

    returns:
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: Training data and labels and
        test data and labels of pytorch MNIST like dataset.
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    TR_MNIST = dataset_loader_func(root=dir_name, train=True, download=True, transform=None)
    TS_MNIST = dataset_loader_func(root=dir_name, train=False, download=True, transform=None)
    TR_X = TR_MNIST.train_data.type(torch.float).reshape(-1, 1, 28, 28).to(device)
    TR_Y = torch.nn.functional.one_hot(TR_MNIST.train_labels).to(device)
    TS_X = TS_MNIST.test_data.type(torch.float).reshape(-1, 1, 28, 28).to(device)
    TS_Y = torch.nn.functional.one_hot(TS_MNIST.test_labels).to(device)
    return (TR_X, TR_Y), (TS_X, TS_Y)
