from pathlib import Path
from itertools import accumulate
from functools import lru_cache, cached_property
from bisect import bisect
import logging
import random
from typing import Sequence, Optional, List, Literal

from torch.utils.data import Dataset, Sampler
from fastparquet import ParquetFile
import pandas as pd
import numpy as np

from .distributed_data_utils import process_record


_MAX_OPEN_RG = 1
_logger = logging.getLogger("event_seq")


class _GroupSampler(Sampler):
    def __init__(
        self,
        group_lens: Sequence[int],
        use_groups: Optional[List[int]] = None,
        shuffle: bool = False,
    ):
        self._gr_idx = (
            use_groups[:] if use_groups is not None else list(range(len(group_lens)))
        )
        self._gls = group_lens
        self._gr_bounds = [0, *accumulate(self._gls)]
        self._len = sum(self._gls[i] for i in self._gr_idx)
        self._shuffle = shuffle
        _logger.debug("group lens: %s, groups used: %s", self._gls, self._gr_idx)

    def __iter__(self):
        idx = []

        if self._shuffle:
            random.shuffle(self._gr_idx)

        _logger.debug("groups order: %s", self._gr_idx)
        for i in self._gr_idx:
            gr_len = self._gls[i]
            first_gr_idx = self._gr_bounds[i]
            intra_group_idx = list(range(first_gr_idx, first_gr_idx + gr_len))
            if self._shuffle:
                random.shuffle(intra_group_idx)
            idx.extend(intra_group_idx)

        return iter(idx)

    def __len__(self):
        return self._len


class ParquetDataset(Dataset):
    def __init__(self, data_path: Path, data_conf):
        self._data_path = data_path.resolve()
        self.data_conf = data_conf
        self._pf = ParquetFile(data_path.as_posix())

        self._rg_lens = [rg.num_rows for rg in self._pf.row_groups]
        _logger.debug("row group lens: %s", self._rg_lens)
        self._len = sum(self._rg_lens)
        self._rg_index = [*accumulate(self._rg_lens), 0]

    def __getitem__(self, idx: int):
        # assert idx >= self._rg_index[self._last_rg_num],\
        # "row groups can be accessed in sequential order only"

        rg_idx = bisect(self._rg_index[:-1], idx)
        idx_inside_rg = idx - self._rg_index[rg_idx - 1]
        df_rg = self._get_rg(rg_idx)
        return df_rg.iloc[idx_inside_rg].copy()

    def get_sampler(self, shuffle: bool = False):
        return _GroupSampler(self._rg_lens, shuffle=shuffle)

    def get_train_val_samplers(
        self,
        approx_val_fraction: float,
        shuffle: Literal["none", "train", "val", "both"] = "train",
    ):
        train_groups = []
        val_groups = []
        target_val_len = int(approx_val_fraction * len(self))

        n_groups = len(self._rg_lens)
        rg_idx_shuffled = random.Random(self.data_conf.client_list_shuffle_seed).sample(
            range(n_groups), n_groups
        )

        for i in rg_idx_shuffled:
            rg_len = self._rg_lens[i]
            if rg_len > target_val_len:
                train_groups.append(i)
            else:
                val_groups.append(i)
                target_val_len -= rg_len

        actual_val_size = sum(self._rg_lens[i] for i in val_groups)
        msg = f"Actual val fraction is {actual_val_size / len(self):.4f}"
        if actual_val_size == 0:
            _logger.warning(msg)
        else:
            _logger.info(msg)
        return _GroupSampler(
            self._rg_lens, use_groups=train_groups, shuffle=shuffle in {"train", "both"}
        ), _GroupSampler(
            self._rg_lens, use_groups=val_groups, shuffle=shuffle in {"val", "both"}
        )

    @lru_cache(maxsize=_MAX_OPEN_RG)
    def _get_rg(self, rg_idx):
        _logger.debug("Loading new parquet row group (could last for tens of seconds)")
        _logger.debug("Loading group %s", rg_idx)
        return self._pf[rg_idx].to_pandas()

    def __hash__(self):  # for `lru_cache` to work
        return hash(self._data_path)

    def __eq__(self, other):  # for `lru_cache` to work
        return self._data_path == other._data_path

    def __len__(self):
        return self._len


class TxnParquetDataset(ParquetDataset):
    def __init__(self, data_conf, test=False):
        self.conf = data_conf
        data_path = Path(data_conf.test_path if test else data_conf.train_path)

        super().__init__(data_path, data_conf)

        self.embeddings = list(data_conf.features.embeddings.keys())
        self.feature_keys = self.embeddings + list(
            data_conf.features.numeric_values.keys()
        )

    def __getitem__(self, idx: int):
        rec = super().__getitem__(idx)
        rec = {k: np.asanyarray(v) for k, v in rec.to_dict().items()}
        rec = process_record(rec, self.conf, self.feature_keys, self.embeddings)
        return rec
