from functools import lru_cache
from typing import Union, Tuple, Sequence
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

## KPI series 
class KPISeries(object):
    def __init__(self, value, timestamp, truth=None, label=None, missing=None):
        self._value = np.asarray(value, np.float32)
        self._timestamp = np.asarray(timestamp, np.int64)
        self._truth = np.asarray(truth, np.int) if truth is not None else np.zeros(value.shape, dtype=np.int)
        self._label = np.asarray(label, np.int) if label is not None else np.zeros(value.shape, dtype=np.int)
        self._missing = np.asarray(missing, np.int) if missing is not None else np.zeros(value.shape, dtype=np.int)
        self._label[self._missing == 1] = 0
        self._truth[self._missing == 1] = 0

        self._check_shape()

        def __update_with_index(__index):
            self._timestamp = self.timestamp[__index]
            self._label = self.label[__index]
            self._truth = self.truth[__index]
            self._missing = self.missing[__index]
            self._value = self.value[__index]

        # check interval and add missing
        __update_with_index(np.argsort(self.timestamp))
        __update_with_index(np.unique(self.timestamp, return_index=True)[1])
        intervals = np.diff(self.timestamp)
        interval = np.min(intervals)
        assert interval > 0, "interval must be positive:{}".format(interval)
        if not np.max(intervals) == interval:
            index = (self.timestamp - self.timestamp[0]) // interval
            new_timestamp = np.arange(self.timestamp[0], self.timestamp[-1] + 1, interval)
            assert new_timestamp[-1] == self.timestamp[-1] and new_timestamp[0] == self.timestamp[0]
            assert np.min(np.diff(new_timestamp)) == interval
            new_value = np.ones(new_timestamp.shape, dtype=np.float32) * self.missing_value
            new_value[index] = self.value
            new_label = np.zeros(new_timestamp.shape, dtype=np.int)
            new_label[index] = self.label
            new_truth = np.zeros(new_timestamp.shape, dtype=np.int)
            new_truth[index] = self.truth
            new_missing = np.ones(new_timestamp.shape, dtype=np.int)
            new_missing[index] = self.missing
            self._timestamp, self._value, self._label, self._truth, self._missing = new_timestamp, new_value, new_label, new_truth, new_missing
            self._check_shape()

    def _check_shape(self):
        # check shape
        assert self._value.shape == self._timestamp.shape == self._missing.shape == self._label.shape == self._truth.shape, \
        "data shape mismatch, value:{}, timestamp:{}, missing:{}, label:{}, truth:{}".format(
                self._value.shape, self._timestamp.shape, self._missing.shape, self._label.shape, self._truth.shape)
        
      
    @property
    def value(self):
        return self._value

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def label(self):
        return self._label

    @property
    def truth(self):
        return self._truth

    @property
    def missing(self):
        return self._missing

    @property
    def time_range(self):
        from datetime import datetime
        return datetime.fromtimestamp(np.min(self.timestamp)), datetime.fromtimestamp(np.max(self.timestamp))

    @property
    def length(self):
        return np.size(self.value, 0)

    @property
    def abnormal(self):
        return np.logical_or(self.missing, self.label).astype(np.int)

    @property
    def missing_rate(self):
        return float(np.count_nonzero(self.missing)) / float(self.length)

    @property
    def anormaly_rate(self):
        # return float(np.count_nonzero(self.label)) / float(self.length)
        return float(np.count_nonzero(self.truth)) / float(self.length)

    @property
    def missing_value(self):
        return 0.0

    @lru_cache()
    def normalize(self, mean=None, std=None, return_statistic=False):
        """
        """
        mean = np.mean(self.value) if mean is None else mean
        std = np.std(self.value) if std is None else std
        normalized_value = (self.value - mean) / np.clip(std, 1e-4, None)
        target = KPISeries(value=normalized_value, timestamp=self.timestamp, label=self.label, truth=self.truth, missing=self.missing)
        if return_statistic:
            return target, mean, std
        else:
            return target

    def split(self, radios):
        """
        :param radios: radios of each part, eg. (0.5, 0.5)
        :return: tuple of DataSets
        """
        if np.asarray(radios).ndim == 1:
            radios = radios  # type: Tuple[float, float]
            assert abs(1.0 - sum(radios)) < 1e-4
            split = np.asarray(np.cumsum(np.asarray(radios, np.float64)) * self.length, np.int)
            split[-1] = self.length
            split = np.concatenate([[0], split])
            result = []
            for l, r in zip(split[:-1], split[1:]):
                result.append(KPISeries(value=self.value[l:r], timestamp=self.timestamp[l:r], label=self.label[l:r], truth=self.truth[l:r], missing=self.missing[l:r]))
        else:
            raise ValueError("split radios in wrong format: {}".format(radios))
        ret = tuple(result) 
        return ret



    @lru_cache()
    def label_sampling(self, sampling_rate: float = 1.):
        """
        sampling label by segments
        :param sampling_rate: keep at most sampling_rate labels
        :return:
        """
        sampling_rate = float(sampling_rate)
        assert 0. <= sampling_rate <= 1., "sample rate must be in [0, 1]: {}".format(sampling_rate)
        if sampling_rate == 1.:
            return self
        elif sampling_rate == 0.:
            return KPISeries(value=self.value, timestamp=self.timestamp, truth=self.truth, label=None, missing=self.missing)
        else:
            target = np.count_nonzero(self.label) * sampling_rate
            label = np.copy(self.label).astype(np.int8)
            anormaly_start = np.where(np.diff(label) == 1)[0] + 1
            if label[0] == 1:
                anormaly_start = np.concatenate([[0], anormaly_start])
            anormaly_end = np.where(np.diff(label) == -1)[0] + 1
            if label[-1] == 1:
                anormaly_end = np.concatenate([anormaly_end, [len(label)]])

            x = np.arange(len(anormaly_start))
            np.random.shuffle(x)

            for i in range(len(anormaly_start)):
                idx = np.asscalar(np.where(x == i)[0])
                label[anormaly_start[idx]:anormaly_end[idx]] = 0
                if np.count_nonzero(label) <= target:
                    break
            return KPISeries(value=self.value, timestamp=self.timestamp, truth=self.truth, label=label, missing=self.missing)
    
    def __len__(self):
        return len(self.timestamp)


class KpiFrameDataset(Dataset):
    def __init__(self, kpi, frame_size, missing_injection_rate=0.0):
        self._kpi = kpi
        self._frame_size = frame_size

        self._strided_value = self.to_frames(kpi.value, frame_size)
        self._strided_abnormal = self.to_frames(kpi.abnormal, frame_size)
        self._strided_missing = self.to_frames(kpi.missing, frame_size)
        self._strided_label = self.to_frames(kpi.label, frame_size)
        self._missing_injection_rate = missing_injection_rate
        self._missing_value = kpi.missing_value

    def __len__(self):
        return np.size(self._strided_value, 0)

    def __getitem__(self, item):
        value = np.copy(self._strided_value[item]).astype(np.float32)
        normal = 1 - np.copy(self._strided_abnormal[item]).astype(np.int)
        label = np.copy(self._strided_label[item]).astype(np.int)

        _missing_injection(value, normal=normal, label=label, missing_value=self._missing_value,
                           missing_injection_rate=self._missing_injection_rate)
        return value.astype(np.float32), normal.astype(np.float32)

    @staticmethod
    def to_frames(array, frame_size: int = 120):
        # noinspection PyProtectedMember
        from numpy.lib.stride_tricks import as_strided
        array = as_strided(array, shape=(np.size(array, 0) - frame_size + 1, frame_size),
                           strides=(array.strides[-1], array.strides[-1]))
        return array


def _missing_injection(value, normal, label, missing_value, missing_injection_rate):
    injected_missing = np.random.binomial(1, missing_injection_rate, np.shape(value[normal == 1]))
    normal[normal == 1] = 1 - injected_missing
    value[np.logical_and(normal == 0, label == 0)] = missing_value
    return value, normal


class _IndexSampler(object):
    def __init__(self, length, shuffle, drop_last, batch_size):
        self.idx = np.arange(length)
        if shuffle:
            np.random.shuffle(self.idx)
        self.pos = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.length = length

    def next(self):
        if self.pos + self.batch_size <= self.length:
            data = self.idx[self.pos: self.pos + self.batch_size]
        elif self.pos >= self.length:
            raise StopIteration()
        elif self.drop_last:
            raise StopIteration()
        else:
            data = self.idx[self.pos:]
        self.pos += self.batch_size
        return data


class KpiFrameDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.index_sampler = None  # type: _IndexSampler
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def __next__(self):
        return tuple(torch.from_numpy(_) for _ in self.dataset[self.index_sampler.next()])

    def __iter__(self):
        self.index_sampler = _IndexSampler(length=len(self.dataset), shuffle=self.shuffle, drop_last=self.drop_last,
                                           batch_size=self.batch_size)
        return self

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def _test_index_sampler():
    sampler = _IndexSampler(100, True, True, 11)
    try:
        while True:
            print(sampler.next())
    except StopIteration:
        pass
