from src.loader.mnist_loader import load_mnist
from src.model.resnet18 import ResNet18
from src.model.vgg import vgg8b
from src.trainer.backpropagation import train_backpropagation
from src.trainer.sigprop import train_sigprop


def main():
    TR_X_mnist, TR_Y_mnist, TS_X_mnist, TS_Y_mnist = load_mnist()
    TR_X_mnist.shape, TR_Y_mnist.shape, TS_X_mnist.shape, TS_Y_mnist.shape
    model = vgg8b(num_classes=10)
    train_sigprop(model, (TR_X_mnist, TR_Y_mnist), epochs=2)


if __name__ == '__main__':
    main()
