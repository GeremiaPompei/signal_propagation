import os

import torch
from torchvision import datasets


def load_mnist(device='cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function able to download MNIST dataset and return it.

    returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Training data and labels and test data and labels of MNIST dataset.
    """
    mnist_dir = 'MNIST/'
    if not os.path.exists(mnist_dir):
        os.mkdir(mnist_dir)
    TR_MNIST = datasets.MNIST(root=f'{mnist_dir}', train=True, download=True, transform=None)
    TS_MNIST = datasets.MNIST(root=f'{mnist_dir}', train=False, download=True, transform=None)
    return TR_MNIST.train_data.type(torch.float).reshape(-1, 1, 28, 28).to(device), torch.nn.functional.one_hot(
        TR_MNIST.train_labels).to(device), \
           TS_MNIST.test_data.type(torch.float).reshape(-1, 1, 28, 28).to(device), torch.nn.functional.one_hot(
        TS_MNIST.test_labels).to(device)
