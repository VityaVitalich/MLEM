# coding: utf-8
import logging

from torch.utils.data import Dataset
import torch
import numpy as np

logger = logging.getLogger(__name__)


class SplittingDataset(Dataset):
    def __init__(self, base_dataset, splitter):
        self.base_dataset = base_dataset
        self.splitter = splitter

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row['feature_arrays']
        local_date = row['event_time']

        indexes = self.splitter.split(local_date)
        data = [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]
        return data

class TargetEnumeratorDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]
        data = [(x, idx) for x in row]
        return data

class ConvertingTrxDataset(Dataset):
    def __init__(self, delegate, style='map', with_target=True):
        self.delegate = delegate
        self.with_target = with_target
        if hasattr(delegate, 'style'):
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
        if type(item) is list:
            return [self._one_item(t) for t in item]
        else:
            return self._one_item(item)

    def _one_item(self, item):
        if self.with_target:
            x, y = item
            x = {k: torch.from_numpy(self.to_torch_compatible(v)) for k, v in x.items()}
            return x, y
        else:
            item = {k: torch.from_numpy(self.to_torch_compatible(v)) for k, v in item.items()}
            return item

    @staticmethod
    def to_torch_compatible(a):
        if a.dtype == np.int8:
            return a.astype(np.int16)
        return a

class DropoutTrxDataset(Dataset):
    def __init__(self, dataset: Dataset, trx_dropout, seq_len, with_target=True):
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
        if type(item) is list:
            return [self._one_item(t) for t in item]
        else:
            return self._one_item(item)

    def _one_item(self, item):
        if self.with_target:
            x, y = item
        else:
            x = item

        seq_len = len(next(iter(x.values())))

        
        if self.trx_dropout > 0 and seq_len > 0:
            idx = np.random.choice(seq_len, size=int(seq_len * (1 - self.trx_dropout)), replace=False)
            idx = np.sort(idx)
        else:
            idx = np.arange(seq_len)

        idx = idx[-self.max_seq_len:]
        new_x = {k: v[idx] for k, v in x.items()}

        if self.with_target:
            return new_x, y
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

        feature_arrays = row['feature_arrays']
        local_date = row['event_time']

        data = []
        for splitter in self.splitters:
            indexes = splitter.split(local_date)
            data += [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]
        return data