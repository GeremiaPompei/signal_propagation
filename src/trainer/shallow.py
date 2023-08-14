import torch

from src.trainer.backpropagation import BackpropagationTrainer


class ShallowTrainer(BackpropagationTrainer):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = False
