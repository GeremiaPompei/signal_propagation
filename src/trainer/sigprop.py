from typing import Callable

import torch

from src.trainer.trainer import Trainer


def get_leaf_layers(module: torch.nn.Module):
    children = list(module.children())
    if not children:
        return [module]
    leaves = []
    for layer in children:
        leaves.extend(get_leaf_layers(layer))
    return leaves


class SigpropTrainer(Trainer):
    def __init__(
            self,
            model: torch.nn.Module,
            device: str = 'cpu',
            precision: torch.dtype = None,
            inner_layer_distance_function: Callable = None
    ):
        super().__init__(model, device, precision)
        self.layers = None
        self.output_embedding_layer = None
        if inner_layer_distance_function is None:
            self.inner_layer_distance_function = lambda h_n, t_n: (h_n - t_n).pow(2).mean()
        else:
            self.inner_layer_distance_function = inner_layer_distance_function

    def __initialize__(self):
        if self.layers is None:
            self.layers = list(self.model.children())

    def __initialize_output_embedding_layer(self, h_n, input_features):
        _, self.dim_c, self.dim_w, self.dim_h = h_n.shape
        self.output_embedding_layer = torch.nn.Linear(input_features, self.dim_c * self.dim_w * self.dim_h) \
            .to(self.device) \
            .to(self.precision)

    def train_mb(self, TR_X_MB: torch.Tensor, TR_Y_MB: torch.Tensor):
        self.__initialize__()
        if hasattr(self.model, 'preprocess'):
            h, t = self.model.preprocess(TR_X_MB), self.model.preprocess(TR_Y_MB)
        else:
            h, t = TR_X_MB, TR_Y_MB
        for i, layer in enumerate(self.layers):
            self.optim.zero_grad()
            h.requires_grad, t.requires_grad = True, True
            with torch.autocast(device_type=self.device, dtype=self.precision):
                if i > 0:
                    if type(layer) == torch.nn.Linear:
                        h = h.view(h.shape[0], -1)
                        t = t.view(t.shape[0], -1)
                    h_n, t_n = layer(torch.cat((h, t))).tensor_split(2)
                else:
                    h_n = layer(h)
                    if self.output_embedding_layer is None:
                        self.__initialize_output_embedding_layer(h_n, TR_Y_MB.shape[-1])
                    t_n = self.output_embedding_layer(t).view(-1, self.dim_c, self.dim_w, self.dim_h)
                if i == len(self.layers) - 1:
                    loss = torch.nn.functional.cross_entropy(h_n, TR_Y_MB)
                else:
                    loss = self.inner_layer_distance_function(h_n, t_n)
            try:
                loss.backward()
                self.optim.step()
                h, t = h_n.detach(), t_n.detach()
            except:
                h, t = h_n, t_n
        return loss.item()
