import torch

from src.trainer.backpropagation import BackpropagationTrainer


class ShallowTrainer(BackpropagationTrainer):
    def __init__(
            self,
            model: torch.nn.Module,
            id_name: str,
            device: str = 'cpu',
            precision: torch.dtype = None,
            filename: str = 'results.json',
            evaluate_accuracy: bool = None,
    ):
        super().__init__(model, id_name, device, precision, filename, evaluate_accuracy)
        for param in list(model.parameters())[:-1]:
            param.requires_grad = False
