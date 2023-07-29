import os

import torch
from torchvision import datasets


def load_mnist(device='cpu') -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Function able to download MNIST dataset and return it.

    returns:
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: Training data and labels and test
        data and labels of MNIST dataset.
    """
    mnist_dir = 'MNIST/'
    if not os.path.exists(mnist_dir):
        os.mkdir(mnist_dir)
    TR_MNIST = datasets.MNIST(root=f'{mnist_dir}', train=True, download=True, transform=None)
    TS_MNIST = datasets.MNIST(root=f'{mnist_dir}', train=False, download=True, transform=None)
    TR_X = TR_MNIST.train_data.type(torch.float).reshape(-1, 1, 28, 28).to(device)
    TR_Y = torch.nn.functional.one_hot(TR_MNIST.train_labels).to(device)
    TS_X = TS_MNIST.test_data.type(torch.float).reshape(-1, 1, 28, 28).to(device)
    TS_Y = torch.nn.functional.one_hot(TS_MNIST.test_labels).to(device)
    return (TR_X, TR_Y), (TS_X, TS_Y)
