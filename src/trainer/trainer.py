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
            model: torch.nn.Module,
            id_name: str,
            device: str = 'cpu',
            precision: torch.dtype = None,
            filename: str = 'results.json',
            lr: float = 5e-4,
    ):
        self.id_name = id_name
        self.device = device
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lrs = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=64)
        self.precision = precision
        self.base_dir = 'results'
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        self.filename = f'{self.base_dir}/{filename}'

    @abstractmethod
    def train_mb(self, TR_X_MB: torch.Tensor, TR_Y_MB: torch.Tensor):
        pass

    def __write_record_in_file(
            self,
            tr_perf: float,
            ts_perf: float,
            mean_tr_time: float,
            mean_tr_memory_usage: float,
            tr_accuracy: float = None,
            ts_accuracy: float = None,
    ):
        results = {}
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as fr:
                results = json.load(fr)
        results[self.id_name] = dict(
            tr_perf=tr_perf,
            ts_perf=ts_perf,
            mean_tr_time=mean_tr_time,
            mean_tr_memory_usage=mean_tr_memory_usage
        )
        if tr_accuracy is not None:
            results[self.id_name]['tr_accuracy'] = tr_accuracy
        if ts_accuracy is not None:
            results[self.id_name]['ts_accuracy'] = ts_accuracy
        with open(self.filename, 'w') as fw:
            json.dump(results, fw)

    def __call__(
            self,
            TR_SET,
            TS_SET,
            epochs: int = 64,
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
                else torch.cuda.memory_allocated()
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

        tr_accuracy, ts_accuracy = None, None
        for dataset_label, dataset in [
            ('tr_set', TR_SET),
            ('ts_set', TS_SET)
        ]:
            acc_counter, acc_len = 0, 0
            for X_MB, Y_MB in tqdm(dataset, desc=f'compute accuracy of {dataset_label}'):
                counter = (self.model(X_MB).argmax(-1) - Y_MB.argmax(-1) == 0).float()
                acc_counter += sum(counter).item()
                acc_len += len(counter)
            if dataset_label == 'tr_set':
                tr_accuracy = acc_counter / acc_len
            elif dataset_label == 'ts_set':
                ts_accuracy = acc_counter / acc_len
        log.info(f'tr_accuracy: {tr_accuracy:>10.6f} - ts_accuracy: {ts_accuracy:>10.6f}')

        self.__write_record_in_file(
            tr_loss,
            ts_loss,
            sum(tr_times) / len(tr_times),
            sum(tr_memory) / len(tr_memory),
            tr_accuracy=tr_accuracy,
            ts_accuracy=ts_accuracy,
        )
        return self.model
