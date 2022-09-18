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
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        batch_size=256,
    ):
        self.df = None
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = self.input_width + self.shift
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]
        self.batch_size = batch_size
        self.scale_coef = 1000.0
        self.bias = 0.0
        self.clip_value = 2.0

    def load_tsv(
        self,
        tsv_filename,
        input_column="open",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        verbose=1,
    ):
        self.df = pd.read_csv(
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
        return self.load_df(
            self.df, input_column, train_ratio, val_ratio, test_ratio, verbose
        )

    def load_df(
        self,
        df: pd.core.frame.DataFrame,
        input_column="open",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        verbose=1,
    ):
        """
        Преобразует dataframe в обучающую выборку
        dataframe - Pandas Dataframe с колонками
        ["date", "time", "open", "high", "low",
            "close", "tickvol", "spread", "real_volume"]
        input_column - колонка с основными данными
        """
        self.df = df
        self.add_hours()
        df_size = self.df[input_column].size
        train_size = int(df_size * train_ratio)
        val_size = int(df_size * val_ratio)
        test_size = int(df_size * test_ratio)
        train_slice = slice(-train_size, None)
        val_slice = slice(-(train_size + val_size), -train_size)
        test_slice = slice(
            -(train_size + val_size + test_size), -(train_size + val_size)
        )
        self.train_rates = self.df[input_column][train_slice]
        self.val_rates = self.df[input_column][val_slice]
        self.test_rates = self.df[input_column][test_slice]
        self.train_hours = self.df['hours'][train_slice]
        self.val_hours = self.df['hours'][val_slice]
        self.test_hours = self.df['hours'][test_slice]
        if verbose == 1:
            print(self.__sizes__())
            print(self.__repr__())
        return True

    def __sizes__(self):
        return "\n".join(
            [
                f"Размер train: {len(self.train_rates)}",
                f"Размер validation: {len(self.val_rates)}",
                f"Размер test: {len(self.test_rates)}",
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

    def add_hours(self):
        times = self.df['date'].str.cat(self.df['time'], sep=' ')
        tics_in_hour = 60*60*10**9
        # Добавляем колонку hour
        self.df['hours'] = pd.to_datetime(times).apply(
            lambda x: x.value//tics_in_hour % 24
        )

    def make_dataset(self, rates, hours):
        # input_data = rates[:-self.input_width]
        # targets = rates[self.input_width:]
        data = np.c_[self.transform(rates), hours[:-1]]
        drates = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,
        )
        drates = drates.map(self.split_window)
        # dhours = dhours.map(self.split_window)
        return drates

    def make_input(self, rates, hours):
        data = np.c_[self.transform(rates), hours[:-1]]
        return tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.input_width,
            sequence_stride=1,
            shuffle=False,
        )

    def make_output(self, output_data):
        d = self.inverse_transform(output_data)
        return d

    def split_window(self, databatch):
        inputs = databatch[:, self.input_slice, :]
        labels = databatch[:, self.labels_slice, 0]
        inputs.set_shape([None, self.input_width, None])
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
        print(f"std={std}, mean={mean}")
        return ds

    def inverse_transform(self, output_data):
        d = (output_data + self.bias)*self.scale_coef
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

    @ property
    def train(self):
        return self.make_dataset(self.train_rates, self.train_hours)

    @ property
    def val(self):
        return self.make_dataset(self.val_rates, self.val_hours)

    @ property
    def test(self):
        return self.make_dataset(self.test_rates, self.test_hours)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def show_shapes(self):
        for example_inputs, example_labels in self.train.take(1):
            print(
                f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(
                f'Labels shape (batch, time, features): {example_labels.shape}')
