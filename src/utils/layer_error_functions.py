from abc import ABC, abstractmethod

import torch


class LayerErrorFunction(ABC):
    @abstractmethod
    def __call__(self, h_n: torch.Tensor, t_n: torch.Tensor) -> float:
        pass


class L2LEF(LayerErrorFunction):
    def __call__(self, h_n: torch.Tensor, t_n: torch.Tensor) -> float:
        return (h_n - t_n).abs().norm(2).mean()


class DotProdLEF(LayerErrorFunction):
    def __call__(self, h_n: torch.Tensor, t_n: torch.Tensor) -> float:
        h = h_n.view(h_n.shape[0], 1, -1)
        t = t_n.view(t_n.shape[0], 1, -1)
        num = (h @ t.transpose(1, 2)).squeeze()
        den = h.squeeze().norm(dim=-1) * t.squeeze().norm(dim=-1)
        prod = num / den
        mean_res = prod.abs().mean()
        res = - mean_res.log()
        res = torch.clip(res, -100, 0)
        return res
