import functools
import operator
from collections import defaultdict
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ..data_load import split_strategy
from .data_utils import prepare_data, prepare_test_data
from .splitting_dataset import (
    ConvertingTrxDataset,
    DropoutTrxDataset,
    SplittingDataset,
    SberSplittingDataset,
    TargetEnumeratorDataset,
)

# from .distributed_data_utils import prepare_data_dist, prepare_test_data_dist
from .parquet_ds import TxnParquetDataset


def create_data_loaders(conf, **kwargs):
    if conf.get("load_distributed", False):
        return _create_data_loaders_distributed(conf, **kwargs)
    else:
        return _create_data_loaders_basic(conf, **kwargs)


def _create_data_loaders_basic(conf, supervised=True, pinch_test=False):
    train_data, valid_data, test_data = prepare_data(conf, supervised, pinch_test)

    dataset_class = SberSplittingDataset if hasattr(conf, "sber") else SplittingDataset

    train_dataset = dataset_class(
        train_data,
        split_strategy.create(**conf.train.split_strategy),
        conf.features.target_col,
        conf.track_metric,
    )
    train_dataset = TargetEnumeratorDataset(train_dataset)
    train_dataset = ConvertingTrxDataset(train_dataset)
    train_dataset = DropoutTrxDataset(
        train_dataset, trx_dropout=conf.train.dropout, seq_len=conf.train.max_seq_len
    )
    if conf.use_constant_pad:
        collate_func = functools.partial(
            collate_splitted_rows_constant, max_len=conf.train.max_seq_len
        )
    else:
        collate_func = collate_splitted_rows

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=collate_func,
        num_workers=conf.train.num_workers,
        batch_size=conf.train.batch_size,
    )

    valid_dataset = dataset_class(
        valid_data,
        split_strategy.create(**conf.val.split_strategy),
        conf.features.target_col,
    )
    valid_dataset = TargetEnumeratorDataset(valid_dataset)
    valid_dataset = ConvertingTrxDataset(valid_dataset)
    valid_dataset = DropoutTrxDataset(
        valid_dataset, trx_dropout=0.0, seq_len=conf.val.max_seq_len
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        collate_fn=collate_func,
        num_workers=conf.val.num_workers,
        batch_size=conf.val.batch_size,
    )

    test_dataset = dataset_class(
        test_data,
        split_strategy.create(**conf.test.split_strategy),
        conf.features.target_col,
    )
    test_dataset = TargetEnumeratorDataset(test_dataset)
    test_dataset = ConvertingTrxDataset(test_dataset)
    test_dataset = DropoutTrxDataset(
        test_dataset, trx_dropout=0.0, seq_len=conf.test.max_seq_len
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        collate_fn=collate_func,
        num_workers=conf.test.num_workers,
        batch_size=conf.test.batch_size,
    )

    if pinch_test:
        return train_loader, valid_loader, test_loader
    return train_loader, valid_loader


def create_test_loader(conf):
    print("Load distributed", conf.get("load_distributed", False))
    # return create_test_loader_basic(conf)
    if conf.get("load_distributed", False):
        return create_test_loader_distributed(conf)
    else:
        return create_test_loader_basic(conf)


def create_test_loader_basic(conf):
    test_data = prepare_test_data(conf)

    dataset_class = SberSplittingDataset if hasattr(conf, "sber") else SplittingDataset

    test_dataset = dataset_class(
        test_data,
        split_strategy.create(**conf.test.split_strategy),
        conf.features.target_col,
    )
    test_dataset = TargetEnumeratorDataset(test_dataset)
    test_dataset = ConvertingTrxDataset(test_dataset)
    test_dataset = DropoutTrxDataset(
        test_dataset, trx_dropout=0.0, seq_len=conf.test.max_seq_len
    )

    if conf.use_constant_pad:
        collate_func = functools.partial(
            collate_splitted_rows_constant, max_len=conf.train.max_seq_len
        )
    else:
        collate_func = collate_splitted_rows

    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        collate_fn=collate_func,
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

    new_idx = torch.tensor([y[0] for _, y in batch])
    new_y = torch.tensor([y[1] for _, y in batch])

    return PaddedBatch(new_x, lengths), torch.cat(
        [new_idx.unsqueeze(0), new_y.unsqueeze(0)], dim=0
    )


def collate_splitted_rows(batch):
    # flattens samples in list of lists to samples in list
    batch = functools.reduce(operator.iadd, batch)
    return padded_collate(batch)


def collate_splitted_rows_constant(batch, max_len):
    batch = functools.reduce(operator.iadd, batch)
    return padded_collate_constant(batch, max_len)


def padded_collate_constant(batch, max_len):
    new_x_ = defaultdict(list)
    for x, _ in batch:
        for k, v in x.items():
            new_x_[k].append(v)

    lengths = torch.LongTensor([len(e) for e in next(iter(new_x_.values()))])
    new_x = {}
    for k, v in new_x_.items():
        if k == "event_time":
            # pad first seq to desired length
            v[0] = torch.nn.ConstantPad1d((0, max_len - v[0].shape[0]), -1.0)(v[0])

            new_x[k] = torch.nn.utils.rnn.pad_sequence(
                v, batch_first=True, padding_value=-1.0
            )
        else:
            v[0] = torch.nn.ConstantPad1d((0, max_len - v[0].shape[0]), 0.0)(v[0])

            new_x[k] = torch.nn.utils.rnn.pad_sequence(
                v, batch_first=True, padding_value=0.0
            )

    new_idx = torch.tensor([y[0] for _, y in batch])
    new_y = torch.tensor([y[1] for _, y in batch])

    return PaddedBatch(new_x, lengths), torch.cat(
        [new_idx.unsqueeze(0), new_y.unsqueeze(0)], dim=0
    )


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


#### ALPHA DATALOADERS)


def _create_data_loaders_distributed(conf, supervised=True, pinch_test=False):
    data = TxnParquetDataset(conf, test=False)
    train_sampler, val_sampler = data.get_train_val_samplers(conf.valid_size)
    assert conf.train.dropout == 0.0
    data = _pump_my_dataset(data, conf, "train")

    if conf.use_constant_pad:
        collate_func = functools.partial(
            collate_splitted_rows_constant, max_len=conf.train.max_seq_len
        )
    else:
        collate_func = collate_splitted_rows

    train_loader = DataLoader(
        dataset=data,
        sampler=train_sampler,
        collate_fn=collate_func,
        num_workers=conf.train.num_workers,
        batch_size=conf.train.batch_size,
    )
    valid_loader = DataLoader(
        dataset=data,
        sampler=val_sampler,
        collate_fn=collate_func,
        num_workers=conf.val.num_workers,
        batch_size=conf.val.batch_size,
    )
    test_loader = []
    if pinch_test:
        print(
            "Warning. Pinch test not implemented yet for ParquetDataset. Passing placeholder list"
        )
        return train_loader, valid_loader, test_loader
    return train_loader, valid_loader


def create_test_loader_distributed(conf):
    test_data = TxnParquetDataset(conf, test=True)
    sampler = test_data.get_sampler()

    test_data = _pump_my_dataset(test_data, conf, "test")

    if conf.use_constant_pad:
        collate_func = functools.partial(
            collate_splitted_rows_constant, max_len=conf.test.max_seq_len
        )
    else:
        collate_func = collate_splitted_rows

    test_loader = DataLoader(
        dataset=test_data,
        sampler=sampler,
        collate_fn=collate_func,
        num_workers=conf.test.num_workers,
        batch_size=conf.test.batch_size,
    )
    return test_loader


def _pump_my_dataset(dataset, conf, split):
    dataset_type = SplittingDataset
    if hasattr(conf, "sber") and conf.sber is True:
        dataset_type = SberSplittingDataset

    dataset = dataset_type(
        dataset,
        split_strategy.create(**conf[split].split_strategy),
        conf.features.target_col,
    )
    dataset = TargetEnumeratorDataset(dataset)
    dataset = ConvertingTrxDataset(dataset)

    dropout = conf[split].dropout if hasattr(conf[split], "dropout") else 0.0
    dataset = DropoutTrxDataset(
        dataset, trx_dropout=dropout, seq_len=conf[split].max_seq_len
    )
    return dataset
