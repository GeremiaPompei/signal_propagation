from abc import ABC, abstractmethod

import torch


class LayerErrorFunction(ABC):
    @abstractmethod
    def __call__(self, h_n: torch.Tensor, t_n: torch.Tensor) -> float:
        pass


class MSELEF(LayerErrorFunction):
    def __call__(self, h_n: torch.Tensor, t_n: torch.Tensor) -> float:
        return (h_n - t_n).pow(2).mean()


class DotProdLEF(LayerErrorFunction):
    def __call__(self, h_n: torch.Tensor, t_n: torch.Tensor) -> float:
        return (h_n.view(h_n.shape[0], -1) @ t_n.view(t_n.shape[0], -1).T).mean()
