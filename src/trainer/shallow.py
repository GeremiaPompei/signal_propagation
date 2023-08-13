import torch

from src.trainer.backpropagation import BackpropagationTrainer


class ShallowTrainer(BackpropagationTrainer):
    def __init__(self, model: torch.nn.Module, device: str = 'cpu', precision: torch.dtype = None,
                 filename: str = 'results.json', ):
        super().__init__(model, device, precision, filename)
        for param in list(model.parameters())[:-1]:
            param.requires_grad = False
