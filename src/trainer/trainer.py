import json
import os
import time

import torch
from tqdm import tqdm

from src.utils import log
from abc import ABC, abstractmethod
import psutil


class Trainer(ABC):

    def __init__(
            self,
            model:
            torch.nn.Module,
            device: str = 'cpu',
            precision: torch.dtype = None,
            filename: str = 'results.json',
    ):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.lrs = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=64)
        self.precision = precision
        self.base_dir = 'results'
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        self.filename = f'{self.base_dir}/{filename}'

    @abstractmethod
    def train_mb(self, TR_X_MB: torch.Tensor, TR_Y_MB: torch.Tensor):
        pass

    def __write_record_in_file(self, tr_perf: float, ts_perf: float, mean_tr_time: float, mean_tr_memory_usage: float):
        results = {}
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as fr:
                results = json.load(fr)
        results[self.__class__.__name__] = dict(
            tr_perf=tr_perf,
            ts_perf=ts_perf,
            mean_tr_time=mean_tr_time,
            mean_tr_memory_usage=mean_tr_memory_usage
        )
        with open(self.filename, 'w') as fw:
            json.dump(results, fw)

    def __call__(
            self,
            TR_SET,
            TS_SET,
            epochs=64,
    ):
        log.info(self.__class__.__name__)
        tr_loss, ts_loss = -1, -1
        tr_times = []
        tr_memory = []

        for epoch in range(epochs):
            tr_loss_sum, ts_loss_sum = 0, 0

            self.model.train()
            start_time = time.time()
            for TR_X_MB, TR_Y_MB in tqdm(TR_SET, desc=f'{epoch + 1}/{epochs} training'):
                tr_loss_sum += self.train_mb(TR_X_MB, TR_Y_MB)
            end_time = time.time()
            tr_times.append(end_time - start_time)
            tr_memory.append(
                psutil.virtual_memory().used
                if self.device == 'cpu'
                else torch.cuda.max_memory_allocated()
            )
            self.lrs.step()
            tr_loss = tr_loss_sum / len(TR_SET)

            self.model.eval()
            for X_MB, Y_MB in tqdm(TS_SET, desc=f'{epoch + 1}/{epochs} evaluation'):
                with torch.autocast(device_type=self.device, dtype=self.precision):
                    ts_loss_sum += torch.nn.functional.cross_entropy(
                        self.model(X_MB),
                        Y_MB
                    ).item()
            ts_loss = ts_loss_sum / len(TS_SET)

            log.info(f'epoch: {epoch + 1:>4}/{epochs} - tr_loss: {tr_loss:>10.6f} - ts_loss: {ts_loss:>10.6f}')

        self.__write_record_in_file(
            tr_loss,
            ts_loss,
            sum(tr_times) / len(tr_times),
            sum(tr_memory) / len(tr_memory),
        )
        return self.model
