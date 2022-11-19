from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras.backend import dtype
import tensorflow as tf
import pandas as pd
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Dataloader:
    def __init__(
        self,
        input_width,
        label_width,
        shift=1,
        clip_by_value=4.0,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        batch_size=256,
    ):
        self.data = None
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = self.input_width + self.shift
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        self.batch_size = batch_size
        self.scale_coef = 1000.0
        self.bias = 0.0
        self.clip_value = clip_by_value

    def load_tsv(
        self,
        tsv_filename,
        input_column="open",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        verbose=1,
        nrows=0,
    ):
        dataframe = pd.read_csv(
            tsv_filename,
            sep="\t",
            header=0,
            dtype={
                "open": np.float32,
                "close": np.float32,
                "tickvol": np.float32,
                "vol": np.float32,
            },
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
        if nrows > 0:
            if nrows > dataframe[input_column].size:
                nrows = dataframe[input_column].size
            self.data = dataframe[-nrows:]
        else:
            self.data = dataframe

        df_size = self.data[input_column].size
        train_size = int(df_size * train_ratio)
        val_size = int(df_size * val_ratio)
        test_size = int(df_size * test_ratio)
        train_slice = slice(-train_size, None)
        val_slice = slice(-(train_size + val_size), -train_size)
        test_slice = slice(
            -(train_size + val_size + test_size), -(train_size + val_size)
        )

        self.train_df = self.data[input_column].iloc[train_slice]
        self.val_df = self.data[input_column].iloc[val_slice]
        self.test_df = self.data[input_column].iloc[test_slice]
        if verbose == 1:
            print(self.__sizes__())
            print(self.__repr__())
        return True

    def load_df(
        self,
        df: pd.core.frame.DataFrame,  # type: ignore
        input_column="open",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        verbose=1,
    ):
        """
        Преобразует dataframe в обучающую выборку
        dataframe - Pandas Dataframe с колонками 
        ["date", "time", "open", "high", "low", "close", "tickvol", "spread", "real_volume"]
        input_column - колонка с основными данными
        """
        self.data = df
        df_size = self.data[input_column].size
        train_size = int(df_size * train_ratio)
        val_size = int(df_size * val_ratio)
        test_size = int(df_size * test_ratio)
        train_slice = slice(-train_size, None)
        val_slice = slice(-(train_size + val_size), -train_size)
        test_slice = slice(
            -(train_size + val_size + test_size), -(train_size + val_size)
        )
        self.train_df = self.data[input_column][train_slice]
        self.val_df = self.data[input_column][val_slice]
        self.test_df = self.data[input_column][test_slice]
        if verbose == 1:
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
            ]
        )

    def make_dataset(self, data):
        ds = self.transform(data)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=ds,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,
        )
        ds = ds.map(self.split_window)
        return ds

    def make_input(self, data):
        ds = self.transform(data)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=ds,
            targets=None,
            sequence_length=self.input_width,
            sequence_stride=1,
            shuffle=False,
        )
        return ds

    def make_output(self, output_data):
        d = self.inverse_transform(output_data)
        return d

    def split_window(self, databatch):
        inputs = databatch[:, self.input_slice]
        labels = databatch[:, self.labels_slice]
        inputs.set_shape([None, self.input_width])
        labels.set_shape([None, self.label_width])
        return inputs, labels

    def transform(self, data):
        # ds = tf.math.subtract(data, data[:, 0:1])
        # self.first_value = data[0:1]
        # ds = np.diff(data) * self.scale_coef + self.bias
        d = np.diff(data)
        std = np.std(d)
        mean = np.mean(d)
        self.bias = mean
        self.scale_coef = std
        ds = d / std - mean
        ds = np.clip(ds, -self.clip_value, self.clip_value)
        # ds = np.clip(ds, -std * self.clip_value, std * self.clip_value)
        print(f"length={len(data)} std={round(std,6)} mean={round(mean,6)}")
        return ds

    def inverse_transform(self, output_data):
        d = (output_data + self.bias) * self.scale_coef
        # d = (output_data - self.bias) / self.scale_coef
        return d

    def moving_average(self, a, n=1):
        if n == 0:
            return a
        result = np.convolve(a, np.ones((n,)) / float(n), mode="valid")
        return result

    def shift_to_zero(self, data):
        # сдвиг начального значения в ноль
        data = tf.math.subtract(data, data[:, 0:1])
        # data, data[:, self.input_width - 1 : self.input_width, :]
        return data

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)
