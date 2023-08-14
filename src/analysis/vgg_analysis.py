import torch
from src.analysis.analysis import Analysis
from src.loader.fashion_mnist_loader import fashion_mnist_loader
from src.loader.mnist_loader import mnist_loader
from src.model.vgg import vgg8b
from src.trainer.backpropagation import BackpropagationTrainer
from src.trainer.shallow import ShallowTrainer
from src.trainer.sigprop import SigpropTrainer
from src.trainer.trainer import Trainer
from src.utils import set_seed
from src.utils.layer_error_functions import MSELEF, DotProdLEF
from src.utils.select_device import select_device


class VGGAnalysis(Analysis):
    def __init__(self):
        super().__init__('VGG')

    def __call__(self):
        device = select_device()
        precision = torch.bfloat16 if device == 'cpu' else torch.float16

        for data_fn, data_loader in [
            ('vgg_mnist.json', mnist_loader),
            ('vgg_fashion_mnist.json', fashion_mnist_loader)
        ]:
            for id_name, trainer_constructor in [
                ('Shallow', ShallowTrainer),
                ('BP', BackpropagationTrainer),
                ('SP', SigpropTrainer),
            ]:
                set_seed(0)
                TR_SET, TS_SET = data_loader(device=device)
                model = vgg8b(num_classes=10)
                additional_fields = {}
                if 'SP' in id_name:
                    additional_fields['deep_sp'] = True
                trainer: Trainer = trainer_constructor(
                    model,
                    id_name,
                    device=device,
                    precision=precision,
                    filename=data_fn,
                    evaluate_accuracy=True,
                    **additional_fields,
                )
                trainer(TR_SET, TS_SET, epochs=10)


VGGAnalysis()
