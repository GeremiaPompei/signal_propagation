import os
from typing import Callable

import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data.dataloader import default_collate


def pytorch_mnist_dataset_loader(
        dir_name: str,
        dataset_loader_func: Callable,
        batch_size: int = 128,
        device: str = 'cpu',
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
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

    def collate_fn(x):
        X, Y = default_collate(x)
        return X.to(device), Y.squeeze().to(device)

    TR_MNIST, TS_MNIST = [
        torch.utils.data.DataLoader(
            dataset_loader_func(
                root=dir_name,
                train=train,
                download=True,
                transform=transforms.ToTensor(),
                target_transform=torchvision.transforms.Compose([
                    lambda x: torch.LongTensor([x]),
                    lambda x: torch.nn.functional.one_hot(x, 10).to(torch.float),
                ]),
            ),
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn
        ) for train in [False, True]
    ]
    return TR_MNIST, TS_MNIST
