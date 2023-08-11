import torch

from src.loader.fashion_mnist_loader import fashion_mnist_loader
from src.loader.mnist_loader import mnist_loader
from src.model.spike_conv_nn import ConvSpikeNN, SurrogateActivation, SpikeActivation
from src.trainer.backpropagation import BackpropagationTrainer
from src.trainer.shallow import ShallowTrainer
from src.trainer.sigprop import SigpropTrainer
from src.utils import set_seed
from src.utils.select_device import select_device


def main():
    set_seed(0)
    device = select_device()

    TR_SET, TS_SET = mnist_loader(device=device)
    model = ConvSpikeNN(num_classes=10, activation_constructor=SurrogateActivation)
    trainer = SigpropTrainer(model, device=device, precision=torch.bfloat16)

    trainer(TR_SET, TS_SET)


if __name__ == '__main__':
    main()
