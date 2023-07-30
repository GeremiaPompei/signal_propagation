import torch

from src.trainer.backpropagation import BackpropagationTrainer


class ShallowTrainer(BackpropagationTrainer):
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        super().__init__(model, device)
        for param in list(model.parameters())[:-1]:
            param.requires_grad = False
