import torch.optim

from src.eval.accuracy_evaluation import accuracy_evaluate
from src.eval.crossentropy_evaluation import crossentropy_evaluate
from src.loader.fashion_mnist_loader import fashion_mnist_loader
from src.model.resnet18 import ResNet18
from src.model.spike_conv_nn import ConvSpikeNN
from src.model.vgg import vgg8b
from src.trainer.backpropagation import BackpropagationTrainer
from src.trainer.shallow import ShallowTrainer
from src.trainer.sigprop import train_sigprop
from src.utils.select_device import select_device


def main():
    device = select_device()

    TR_SET, TS_SET = fashion_mnist_loader(device=device)
    model = ConvSpikeNN(num_classes=10)
    trainer = BackpropagationTrainer(model, device=device)

    trainer(TR_SET, TS_SET)


if __name__ == '__main__':
    main()
