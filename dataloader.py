import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers.core import Lambda


class Dataloader:
    def __init__(
        self,
        input_width,
        label_width,
        shift=1,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        batch_size=256,
    ):
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
            dtype={
                "open": np.float32,
                "close": np.float32,
                "tickvol": np.float32,
                "vol": np.float32,
            },
            names=["date", "time", "open", "high", "low", "close", "tickvol", "vol"],
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
                f"Индексы входа: {self.input_indices}",
                f"Индексы выхода: {self.label_indices}",
            ]
        )

    def make_dataset(self, data):
        ds = np.array(data, dtype="float32")
        ds = np.diff(ds)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=ds,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,
        )
        # ds = ds.map(self.transform)
        ds = ds.map(self.split_window)
        return ds

    # def transform(self, data):
    #     # data = tf.reshape(databatch, (-1, self.total_window_size))
    #     mean, std = tf.nn.moments(data, axes=[1], keepdims=True)
    #     unbiased = tf.math.subtract(data, mean)
    #     normalized = tf.math.divide(unbiased, std)
    #     return normalized

    # def inverse_transform(self, data, output_data):
    #     mean, std = tf.nn.moments(data, axes=[1], keepdims=True)
    #     denormalized = tf.math.multiply(output_data, std)
    #     biased = tf.math.add(denormalized, mean)
    #     return biased

    def make_input(self, data):
        np_data = np.diff(data)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=np_data,
            targets=None,
            sequence_length=self.input_width,
            sequence_stride=1,
            shuffle=False,
        )
        # ds = ds.map(self.transform)
        return ds

    def make_output(self, input_data, output_data):
        # cumsum = np.cumsum(output_data)
        # return self.inverse_transform(input_data, output_data)
        return output_data

    def split_window(self, databatch):
        inputs = databatch[:, self.input_slice]
        labels = databatch[:, self.labels_slice]
        inputs.set_shape([None, self.input_width])
        labels.set_shape([None, self.label_width])
        return inputs, labels

    def moving_average(self, a, n=1):
        if n == 0:
            return a
        result = np.convolve(a, np.ones((n,)) / float(n), mode="valid")
        return result

    def shift_to_zero(self, data):
        data = tf.math.subtract(
            data, data[:, 0:1, 0:1]
        )  # сдвиг начального значения в ноль
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

