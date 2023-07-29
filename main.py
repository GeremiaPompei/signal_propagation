import torch.optim

from src.eval.accuracy_evaluation import accuracy_evaluate
from src.eval.crossentropy_evaluation import crossentropy_evaluate
from src.loader.mnist_loader import load_mnist
from src.model.resnet18 import ResNet18
from src.model.spike_conv_nn import ConvSpikeNN
from src.model.vgg import vgg8b
from src.trainer.backpropagation import train_backpropagation
from src.trainer.sigprop import train_sigprop
from src.utils.select_device import select_device


def main():
    device = select_device()
    TR_SET, TS_SET = load_mnist(device=device)
    model = ConvSpikeNN(num_classes=10)

    def callback():
        crossentropy_evaluate('tr_set', model, TR_SET)
        accuracy_evaluate('tr_set', model, TR_SET)
        crossentropy_evaluate('ts_set', model, TS_SET)
        accuracy_evaluate('ts_set', model, TS_SET)

    train_backpropagation(
        model,
        TR_SET,
        epochs=64,
        device=device,
        callback=callback,
    )


if __name__ == '__main__':
    main()
