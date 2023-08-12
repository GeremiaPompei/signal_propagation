from torchvision import datasets

from src.loader.pytorch_mnist_dataset_loader import pytorch_mnist_dataset_loader


def fashion_mnist_loader(
        device='cpu',
        batch_size: int = 128,
):
    """
    Function able to download Fashion-MNIST dataset and return it.
    :param device: Accelerator where allocate the dataset.
    :param batch_size: Batch size.

    returns:
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: Training data and labels and test
        data and labels of Fashion-MNIST dataset.
    """
    return pytorch_mnist_dataset_loader(
        dir_name='Fashion-MNIST/',
        dataset_loader_func=datasets.FashionMNIST,
        device=device,
        batch_size=batch_size,
    )
