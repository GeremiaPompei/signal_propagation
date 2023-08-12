import os
from typing import Callable

import torch
from torch.utils.data import Dataset


class MNISTDataLoader(Dataset):

    def __init__(
            self,
            data,
            batch_size=128,
            device: str = 'cpu',
            precision: torch.dtype = torch.float32,
            n_classes: int = 10
    ):
        self.device = device
        self.precision = precision
        X = data.train_data.type(torch.int8)
        perm = torch.randperm(X.shape[0])
        self.X = X[perm].split(batch_size, 0)
        self.Y = data.train_labels.type(torch.int8)[perm].split(batch_size, 0)
        self.n_classes = n_classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].to(self.precision).reshape(-1, 1, 28, 28).to(self.device)
        Y = torch.nn.functional.one_hot(
            self.Y[idx].type(torch.LongTensor), self.n_classes
        ).to(self.precision).to(self.device)
        return X, Y


def pytorch_mnist_dataset_loader(
        dir_name: str,
        dataset_loader_func: Callable,
        batch_size: int = 128,
        device: str = 'cpu',
        precision: torch.dtype = torch.float32,
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
    TR_MNIST = MNISTDataLoader(
        dataset_loader_func(root=dir_name, train=True, download=True, transform=None),
        batch_size=batch_size,
        device=device,
        precision=precision,
    )
    TS_MNIST = MNISTDataLoader(
        dataset_loader_func(root=dir_name, train=False, download=True, transform=None),
        batch_size=batch_size,
        device=device,
        precision=precision,
    )

    return TR_MNIST, TS_MNIST
