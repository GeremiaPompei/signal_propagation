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
    precision = torch.bfloat16 if device == 'cpu' else torch.float16

    for data_label, dataset_loader in [
        ('mnist.json', mnist_loader),
        ('fashion_mnist.json', fashion_mnist_loader)
    ]:
        TR_SET, TS_SET = dataset_loader(device=device)
        for trainer_constructor, surrogate, spike_threshold in [
            (ShallowTrainer, True, 1),
            (BackpropagationTrainer, True, 1),
            (SigpropTrainer, True, 0.5),
            (SigpropTrainer, False, 0.5),
        ]:
            set_seed(0)
            model = ConvSpikeNN(num_classes=10, surrogate=surrogate, spike_threshold=spike_threshold)
            trainer = trainer_constructor(model, device=device, precision=precision, filename=data_label)
            trainer(TR_SET, TS_SET)


if __name__ == '__main__':
    main()
