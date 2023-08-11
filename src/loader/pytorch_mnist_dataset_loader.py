import os
from typing import Callable, Tuple, Any

import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data.dataloader import default_collate


def pytorch_mnist_dataset_loader(
        dir_name: str,
        dataset_loader_func: Callable,
        batch_size: int = 128,
        device: str = 'cpu',
) -> tuple:
    """
    Function able to download pytorch MNIST like dataset and return it.
    :param dir_name: Name of directory where save local dataset.
    :param dataset_loader_func: Function able to download and load dataset.
    :param batch_size: Batch size.
    :param device: Accelerator used to allocate data.

    returns:
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: Training data and labels and
        test data and labels of pytorch MNIST like dataset.
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    TR_MNIST = dataset_loader_func(root=dir_name, train=True, download=True, transform=None)
    TS_MNIST = dataset_loader_func(root=dir_name, train=False, download=True, transform=None)
    TR_SET = \
        TR_MNIST.train_data.type(torch.float).reshape(-1, 1, 28, 28).to(device), \
        torch.nn.functional.one_hot(TR_MNIST.train_labels).to(device)
    TS_SET = \
        TS_MNIST.test_data.type(torch.float).reshape(-1, 1, 28, 28).to(device), \
        torch.nn.functional.one_hot(TS_MNIST.test_labels).to(device)

    TR_X, TR_Y = [x.type(torch.float32).split(batch_size, 0) for x in TR_SET]
    TS_X, TS_Y = [x.type(torch.float32).split(batch_size, 0) for x in TS_SET]

    return tuple(zip(TR_X, TR_Y)), tuple(zip(TS_X, TS_Y))
