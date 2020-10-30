import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers.core import Lambda


class Dataloader:
    def __init__(
        self,
        input_width,
        label_width,
        sample_width,
        shift=1,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        batch_size=256,
    ):
        self.input_width = input_width
        self.label_width = label_width
        self.sample_width = sample_width
        self.shift = shift
        self.total_window_size = self.input_width + self.shift
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        self.batch_size = batch_size
        self.scale = 1000.0

    def load(
        self,
        tsv_filename,
        input_column="open",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        verbose=1,
    ):
        df = pd.read_csv(
            tsv_filename,
            sep="\t",
            header=0,
            dtype={"open": float, "close": float, "tickvol": float, "vol": float,},
            names=[
                "date",
                "time",
                "open",
                "high",
                "low",
                "close",
                "tickvol",
                "vol",
                "spread",
            ],
        )
        df_size = df[input_column].size
        train_len = int(df_size * train_ratio)
        val_len = int(df_size * val_ratio)
        test_len = int(df_size * test_ratio)
        train_slice = slice(0, train_len)
        val_slice = slice(train_len, train_len + val_len)
        test_slice = slice(train_len + val_len, None)
        self.train_df = df[input_column][train_slice]
        self.val_df = df[input_column][val_slice]
        self.test_df = df[input_column][test_slice]
        if verbose == 10:
            print(self.__sizes__())
            print(self.__repr__())
        return True

    def __sizes__(self):
        return "\n".join(
            [
                f"Размер train: {len(self.train_df)}",
                f"Размер validation: {len(self.val_df)}",
                f"Размер test: {len(self.test_df)}",
            ]
        )

    def __repr__(self):
        return "\n".join(
            [
                f"Размер окна: {self.total_window_size}",
                f"Размер входа: {self.input_width}",
                f"Размер выхода: {self.label_width}",
                f"Размер экземпляра: {self.sample_width}",
                f"Индексы входа: {self.input_indices}",
                f"Индексы выхода: {self.label_indices}",
            ]
        )

    def split_window(self, databatch):
        inputs = databatch[:, self.input_slice, :]
        labels = (
            databatch[:, self.labels_slice, :]
            - databatch[:, self.input_width - 1 : self.input_width, :]
        )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def shift_to_zero(self, data):
        data = tf.math.subtract(
            data, data[:, 0:1, 0:1]
        )  # сдвиг начального значения в ноль
        # data, data[:, self.input_width - 1 : self.input_width, :]
        return data

    def make_dataset(self, data, verbose=0):
        ds = self.transform(data, verbose=verbose)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=ds,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,
        )
        # ds = ds.map(self.shift_to_zero)
        ds = ds.map(self.split_window)
        return ds

    def make_input(self, data):
        data = self.transform(data)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.input_width,
            sequence_stride=1,
            shuffle=False,
        )
        # ds = ds.map(self.shift_to_zero)
        # ds = ds.map(self.scale)
        return ds
    
    

    def transform(self, input_data, verbose=1):
        data = np.array(input_data, dtype=np.float32)
        data = self.moving_average(data)
        # data = np.diff(data)
        # self.data_mean = data.mean()
        # self.data_std = data.std()
        # data = (data - self.data_mean) / self.data_std
        data = data[data.shape[0] % self.sample_width :].reshape(
            (-1, self.sample_width)
        )
        # if verbose == 1:
        #     print(f"mean={self.data_mean} std={self.data_std}")
        return data

    def moving_average(self, a, n=5):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

