# coding: utf-8
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import date

logger = logging.getLogger(__name__)
SENTINEL = None


class SplittingDataset(Dataset):
    def __init__(self, base_dataset, splitter, target_col=None, track_metric="roc_auc"):
        self.base_dataset = base_dataset
        self.splitter = splitter
        self.target_col = target_col
        self.track_metric = track_metric

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row["feature_arrays"]
        local_date = row["event_time"]

        indexes = self.splitter.split(local_date)
        data = [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]

        if self.target_col:
            target = row[self.target_col]
            try:
                if self.track_metric == "mse":
                    target = float(target)
                else:
                    target = int(target)
            except (ValueError, TypeError):
                target = -1
            return data, target
        return data


class SberSplittingDataset(SplittingDataset):
    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row["feature_arrays"]
        local_date = row["event_time"]

        indexes = self.splitter.split(local_date)
        data = [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]

        report_date, epk_id = row["index"].split("_")
        for split in data:
            length = len(next(iter(split.values())))
            split["report_date"] = np.array(
                [date.fromisoformat(report_date).toordinal()] * length
            )
            split["epk_id"] = np.array([int(epk_id)] * length)

        if self.target_col:
            target = row[self.target_col]
            try:
                target = int(target)
            except (ValueError, TypeError):
                target = -1
            return data, target
        return data


class TargetEnumeratorDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if self.base_dataset.target_col is SENTINEL:
            row = self.base_dataset[idx]
            data = [(x, idx) for x in row]
        else:
            row, target = self.base_dataset[idx]
            data = [(x, (idx, target)) for x in row]
        return data


class TargetDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row, target = self.base_dataset[idx]
        data = [(x, target) for x in row]
        return data


class ConvertingTrxDataset(Dataset):
    def __init__(self, delegate, style="map", with_target=True):
        self.delegate = delegate
        self.with_target = with_target
        if hasattr(delegate, "style"):
            self.style = delegate.style
        else:
            self.style = style

    def __len__(self):
        return len(self.delegate)

    def __iter__(self):
        for rec in iter(self.delegate):
            yield self._one_item(rec)

    def __getitem__(self, idx):
        item = self.delegate[idx]
        if isinstance(item, list):
            return [self._one_item(t) for t in item]
        else:
            return self._one_item(item)

    def _one_item(self, item):
        if self.with_target:
            x, y = item
            x = {k: torch.from_numpy(self.to_torch_compatible(v)) for k, v in x.items()}
            return x, torch.tensor(y)
        else:
            item = {
                k: torch.from_numpy(self.to_torch_compatible(v))
                for k, v in item.items()
            }
            return item

    @staticmethod
    def to_torch_compatible(a):
        if a.dtype == np.int8:
            return a.astype(np.int16)
        return a


class DropoutTrxDataset(Dataset):
    def __init__(
        self,
        dataset: ConvertingTrxDataset,
        trx_dropout,
        seq_len,
        with_target=True,
    ):
        self.core_dataset = dataset
        self.trx_dropout = trx_dropout
        self.max_seq_len = seq_len
        self.style = dataset.style
        self.with_target = with_target

    def __len__(self):
        return len(self.core_dataset)

    def __iter__(self):
        for rec in iter(self.core_dataset):
            yield self._one_item(rec)

    def __getitem__(self, idx):
        item = self.core_dataset[idx]
        if isinstance(item, list):
            return [self._one_item(t) for t in item]
        else:
            return self._one_item(item)

    def _one_item(self, item):
        x = item[0] if self.with_target else item

        seq_len = len(next(iter(x.values())))

        if self.trx_dropout > 0 and seq_len > 0:
            idx = np.random.choice(
                seq_len, size=int(seq_len * (1 - self.trx_dropout)), replace=False
            )
            idx = np.sort(idx)
        else:
            idx = np.arange(seq_len)

        idx = idx[-self.max_seq_len :]
        new_x = {k: v[idx] for k, v in x.items()}

        if self.with_target:
            return new_x, item[1]
        else:
            return new_x


class SeveralSplittingsDataset(Dataset):
    def __init__(self, base_dataset, splitters):
        self.base_dataset = base_dataset
        self.splitters = splitters

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row["feature_arrays"]
        local_date = row["event_time"]

        data = []
        for splitter in self.splitters:
            indexes = splitter.split(local_date)
            data += [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]
        return data
