from torchvision import datasets

from src.loader.pytorch_mnist_dataset_loader import pytorch_mnist_dataset_loader


def mnist_loader(device='cpu'):
    """
    Function able to download MNIST dataset and return it.
    :param device: Accelerator where allocate the dataset.

    returns:
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: Training data and labels and test
        data and labels of MNIST dataset.
    """
    return pytorch_mnist_dataset_loader(
        dir_name='MNIST/',
        dataset_loader_func=datasets.MNIST,
        device=device,
    )
