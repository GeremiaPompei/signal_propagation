from src.eval.accuracy_evaluation import accuracy_evaluate
from src.loader.mnist_loader import load_mnist
from src.model.resnet18 import ResNet18
from src.model.vgg import vgg8b
from src.trainer.backpropagation import train_backpropagation
from src.trainer.sigprop import train_sigprop
from src.utils.select_device import select_device


def main():
    device = select_device()
    TR_SET, TS_SET = load_mnist(device=device)
    model = vgg8b(num_classes=10)

    def callback():
        accuracy_evaluate('ts_set', model, TR_SET)
        accuracy_evaluate('ts_set', model, TS_SET)

    train_sigprop(
        model,
        TR_SET,
        epochs=2,
        device=device,
        callback=callback
    )


if __name__ == '__main__':
    main()
