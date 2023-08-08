import torch

from src.trainer.backpropagation import BackpropagationTrainer


class ShallowTrainer(BackpropagationTrainer):
    def __init__(self, model: torch.nn.Module, device: str = 'cpu', precision: torch.dtype = torch.float32):
        super().__init__(model, device, precision)
        for param in list(model.parameters())[:-1]:
            param.requires_grad = False
