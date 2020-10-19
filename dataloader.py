import numpy as np
import pandas as pd
import tensorflow as tf


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
                f"Размер экземпляра: {self.sample_width}",
                f"Индексы входа: {self.input_indices}",
                f"Индексы выхода: {self.label_indices}",
            ]
        )

    def split_window(self, databatch):
        inputs = databatch[:, self.input_slice, :]
        labels = databatch[:, self.labels_slice, :]
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data, batch_size=256, verbose=1):
        data = self.transform(data)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=batch_size,
        )
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
        return ds

    def make_output(self, data):
        return self.inverse_transform(data)

    def transform(self, input_data):
        data = np.array(input_data, dtype=np.float32)
        data = np.diff(data)
        self.data_mean = data.mean()
        self.data_std = data.std()
        data = (data - self.data_mean) / self.data_std
        data = data[data.shape[0] % self.sample_width :].reshape(
            (-1, self.sample_width)
        )
        return data

    def inverse_transofrm(self, data):
        output = data * self.data_std + self.data_mean
        return output

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

