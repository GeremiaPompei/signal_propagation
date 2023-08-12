import torch

from src.loader.fashion_mnist_loader import fashion_mnist_loader
from src.loader.mnist_loader import mnist_loader
from src.model.spike_conv_nn import ConvSpikeNN
from src.trainer.backpropagation import BackpropagationTrainer
from src.trainer.shallow import ShallowTrainer
from src.trainer.sigprop import SigpropTrainer
from src.utils import set_seed
from src.utils.select_device import select_device


def main():
    device = select_device()
    precision = torch.bfloat16

    set_seed(0)
    TR_SET, TS_SET = mnist_loader(device=device, precision=precision)
    model = ConvSpikeNN(num_classes=10, surrogate=False)
    trainer = SigpropTrainer(model, device=device, precision=precision)

    trainer(TR_SET, TS_SET)


if __name__ == '__main__':
    main()
