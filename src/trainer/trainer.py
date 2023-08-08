import torch
from src.utils import log
from abc import ABC, abstractmethod


class Trainer(ABC):

    def __init__(self, model: torch.nn.Module, device: str = 'cpu', precision: torch.dtype = torch.float32):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.lrs = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=64)
        self.precision = precision

    @abstractmethod
    def train_mb(self, TR_X_MB: torch.Tensor, TR_Y_MB: torch.Tensor):
        pass

    def __call__(
            self,
            TR_SET: tuple[torch.Tensor, torch.Tensor],
            TS_SET: tuple[torch.Tensor, torch.Tensor],
            epochs=64,
            batch_size=128,
    ):
        self.model = self.model.to(self.precision)
        TR_X, TR_Y = [x.type(self.precision).split(batch_size, 0) for x in TR_SET]
        TS_X, TS_Y = [x.type(self.precision).split(batch_size, 0) for x in TS_SET]

        log.info(self.__class__.__name__)
        for epoch in range(epochs):
            tr_loss_sum, ts_loss_sum = 0, 0

            self.model.train()
            for TR_X_MB, TR_Y_MB in list(zip(TR_X, TR_Y)):
                tr_loss_sum += self.train_mb(TR_X_MB, TR_Y_MB)
            tr_loss = tr_loss_sum / len(TR_X)

            self.model.eval()
            for X_MB, Y_MB in list(zip(TS_X, TS_Y)):
                ts_loss_sum += torch.nn.functional.cross_entropy(self.model(X_MB), Y_MB)
            ts_loss = ts_loss_sum / len(TS_X)

            log.info(f'epoch: {epoch + 1:>4}/{epochs} - tr_loss: {tr_loss:>10.6f} - ts_loss: {ts_loss:>10.6f}')

        return self.model
