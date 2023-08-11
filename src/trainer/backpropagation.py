import torch

from src.trainer.trainer import Trainer


class BackpropagationTrainer(Trainer):
    def train_mb(self, TR_X_MB: torch.Tensor, TR_Y_MB: torch.Tensor):
        self.optim.zero_grad()
        with torch.autocast(device_type=self.device, dtype=self.precision):
            TR_P_MB = self.model(TR_X_MB)
            loss = self.criterion(TR_P_MB, TR_Y_MB)
        loss.backward()
        self.optim.step()
        return loss.item()
