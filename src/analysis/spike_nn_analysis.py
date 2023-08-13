from src.analysis.analysis import Analysis

import torch

from src.loader.fashion_mnist_loader import fashion_mnist_loader
from src.loader.mnist_loader import mnist_loader
from src.model.spike_conv_nn import ConvSpikeNN
from src.trainer.backpropagation import BackpropagationTrainer
from src.trainer.shallow import ShallowTrainer
from src.trainer.sigprop import SigpropTrainer
from src.utils import set_seed
from src.utils.select_device import select_device


class SpikeNNAnalysis(Analysis):
    def __init__(self):
        super().__init__('SpikeNN')

    def __call__(self):
        device = select_device()
        precision = torch.bfloat16 if device == 'cpu' else torch.float16

        for data_fn, data_loader in [
            ('mnist.json', mnist_loader),
            ('fashion_mnist.json', fashion_mnist_loader)
        ]:
            TR_SET, TS_SET = data_loader(device=device)
            for id_name, trainer_constructor, surrogate, spike_threshold in [
                ('Shallow', ShallowTrainer, True, 1),
                ('BP Surrogate', BackpropagationTrainer, True, 1),
                ('SP Surrogate', SigpropTrainer, True, 0.5),
                ('SP Voltage', SigpropTrainer, False, 0.5),
            ]:
                set_seed(0)
                model = ConvSpikeNN(num_classes=10, surrogate=surrogate, spike_threshold=spike_threshold)
                trainer = trainer_constructor(model, id_name, device=device, precision=precision, filename=data_fn)
                trainer(TR_SET, TS_SET)


SpikeNNAnalysis()
