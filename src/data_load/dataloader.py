import functools
import operator
from collections import defaultdict
from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..data_load import split_strategy
from .data_utils import prepare_data, prepare_test_data
from .splitting_dataset import (
    ConvertingTrxDataset,  # TargetDataset
    DropoutTrxDataset,
    SplittingDataset,
    SberSplittingDataset,
    TargetEnumeratorDataset,
)


def create_data_loaders(conf, supervised=True):
    train_data, valid_data = prepare_data(conf, supervised)

    dataset_class = SberSplittingDataset if hasattr(conf, "sber") else SplittingDataset
    train_dataset = dataset_class(
        train_data,
        split_strategy.create(**conf.train.split_strategy),
        conf.features.target_col,
    )
    train_dataset = TargetEnumeratorDataset(train_dataset)
    # train_dataset = TargetDataset(train_dataset)
    train_dataset = ConvertingTrxDataset(train_dataset)
    # не уверен что нам нужна история с дропаутом точек.
    # Но это выглядит неплохой аугментацией в целом
    if conf.train.dropout > 0:
        train_dataset = DropoutTrxDataset(
            train_dataset, trx_dropout=conf.train.dropout, seq_len=conf.train.max_seq_len
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=collate_splitted_rows,
        num_workers=conf.train.num_workers,
        batch_size=conf.train.batch_size,
    )

    valid_dataset = dataset_class(
        valid_data,
        split_strategy.create(**conf.val.split_strategy),
        conf.features.target_col,
    )
    valid_dataset = TargetEnumeratorDataset(valid_dataset)
    # valid_dataset = TargetDataset(valid_dataset)
    valid_dataset = ConvertingTrxDataset(valid_dataset)
    valid_dataset = DropoutTrxDataset(
        valid_dataset, trx_dropout=0.0, seq_len=conf.val.max_seq_len
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        collate_fn=collate_splitted_rows,
        num_workers=conf.val.num_workers,
        batch_size=conf.val.batch_size,
    )

    return train_loader, valid_loader


def create_test_loader(conf):
    test_data = prepare_test_data(conf)

    dataset_class = SberSplittingDataset if hasattr(conf, "sber") else SplittingDataset

    test_dataset = dataset_class(
        test_data,
        split_strategy.create(**conf.test.split_strategy),
        conf.features.target_col,
    )
    test_dataset = TargetEnumeratorDataset(test_dataset)
    # valid_dataset = TargetDataset(valid_dataset)
    test_dataset = ConvertingTrxDataset(test_dataset)
    test_dataset = DropoutTrxDataset(
        test_dataset, trx_dropout=0.0, seq_len=conf.test.max_seq_len
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        collate_fn=collate_splitted_rows,
        num_workers=conf.test.num_workers,
        batch_size=conf.test.batch_size,
    )

    return test_loader


def padded_collate(batch):
    new_x_ = defaultdict(list)
    for x, _ in batch:
        for k, v in x.items():
            new_x_[k].append(v)

    lengths = torch.LongTensor([len(e) for e in next(iter(new_x_.values()))])

    new_x = {}
    for k, v in new_x_.items():
        if k == "event_time":
            new_x[k] = torch.nn.utils.rnn.pad_sequence(
                v, batch_first=True, padding_value=-1.0
            )
        else:
            new_x[k] = torch.nn.utils.rnn.pad_sequence(
                v, batch_first=True, padding_value=0.0
            )
    # new_x = {
    #     k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
    #     for k, v in new_x_.items()
    # }
    new_idx = torch.tensor([y[0] for _, y in batch])
    new_y = torch.tensor([y[1] for _, y in batch])

    return PaddedBatch(new_x, lengths), torch.cat(
        [new_idx.unsqueeze(0), new_y.unsqueeze(0)], dim=0
    )


def collate_splitted_rows(batch):
    # flattens samples in list of lists to samples in list
    batch = functools.reduce(operator.iadd, batch)
    return padded_collate(batch)


class PaddedBatch:
    def __init__(self, payload: Dict[str, torch.Tensor], length: torch.LongTensor):
        self._payload = payload
        self._length = length

    @property
    def payload(self):
        return self._payload

    @property
    def seq_lens(self):
        return self._length

    def __len__(self):
        return len(self._length)

    def to(self, device, non_blocking=False):
        length = self._length.to(device=device, non_blocking=non_blocking)
        payload = {
            k: v.to(device=device, non_blocking=non_blocking)
            for k, v in self._payload.items()
        }
        return PaddedBatch(payload, length)  # type: ignore
