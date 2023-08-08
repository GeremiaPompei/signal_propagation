import torch

from src.trainer.trainer import Trainer


def get_leaf_layers(module: torch.nn.Module, device='cpu'):
    children = list(module.children())
    if not children:
        return [module.to(device)]
    leaves = []
    for layer in children:
        leaves.extend(get_leaf_layers(layer, device=device))
    return leaves


class SigpropTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        super().__init__(model, device)
        self.initialized = False

    def __initialize__(self, TR_X_MB, TR_Y_MB):
        if not self.initialized:
            self.layers = get_leaf_layers(self.model, device=self.device)
            self.dim_c = self.layers[0].out_channels
            _, _, self.dim_w, self.dim_h = TR_X_MB.shape
            self.output_embedding_layer = torch.nn.Linear(TR_Y_MB.shape[1], self.dim_c * self.dim_w * self.dim_h).to(
                self.device)
            self.initialized = True

    def train_mb(self, TR_X_MB: torch.Tensor, TR_Y_MB: torch.Tensor):
        self.__initialize__(TR_X_MB, TR_Y_MB)
        h, t = TR_X_MB, TR_Y_MB
        layers_loss = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                cat_ht = torch.cat((h, t)).squeeze()
                if type(layer) == torch.nn.Linear:
                    cat_ht = cat_ht.view(cat_ht.shape[0], -1)
                h_n, t_n = layer(cat_ht).split(h.shape[0])
            else:
                h_n = layer(h)
                t_n = self.output_embedding_layer(t).view(-1, self.dim_c, self.dim_w, self.dim_h)
            self.optim.zero_grad()
            if i == len(self.layers) - 1:
                loss = torch.nn.functional.cross_entropy(h_n, TR_Y_MB)
            else:
                loss = torch.nn.functional.mse_loss(h_n, t_n)
            try:
                loss.backward()
                self.optim.step()
                h, t = h_n.detach(), t_n.detach()
                layers_loss.append(loss.item())
            except:
                h, t = h_n, t_n
        return torch.Tensor(layers_loss).mean()
