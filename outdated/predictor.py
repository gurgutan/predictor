import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import io
import numpy as np
from numpy import linalg
import pandas as pd
import tensorflow as tf
import pywt
from tensorflow import keras
import random
from os import path
import time
import datetime
import sys
from patterns import (
    conv2D,
    multiConv2D,
    conv1D,
    reslike,
    loc1d,
    multicol_conv1d_block,
    dense_block,
    wave_len,
    lstm_block,
    shifted_mse,
)
from datainfo import DatasetInfo
from tqdm import tqdm
import logging
import pydot
import graphviz
import sklearn.preprocessing as preprocessing
import joblib

# import tflite_runtime.interpreter as tflite
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger = logging.getLogger(__name__)


def roll(a, size, dx=1):
    shape = a.shape[:-1] + (int((a.shape[-1] - size) / dx) + 1, size)
    strides = a.strides + (a.strides[-1] * dx,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def embed(v, min_v, max_v, dim):
    """Возвращает бинарный вектор, длины dim"""
    step_size = dim / (max_v - min_v)
    v = max(min_v, min(max_v - 0.0000001, v))
    n = int((v - min_v) * step_size)
    # result = np.zeros(dim, dtype="float64")
    result = np.full(dim, 0.1, dtype="float64")
    result[n] = 1
    return result


def unembed(n: int, min_v: float, max_v: float, dim: int) -> float:
    step_size = float((max_v - min_v) / dim)
    v = min_v + n * step_size
    return v


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


class Predictor(object):
    def __init__(
        self,
        modelname,
        input_shape=(256, 1),
        output_shape=(8,),
        predict_size=16,
        filters=64,
        kernel_size=4,
        dense_size=32,
    ):
        self.trained = False
        self.interpreter = None
        self.name = modelname
        self.Scaler = preprocessing.StandardScaler()

        if modelname == None:
            self.name = "default"
        else:
            self.name = modelname
        if not path.exists(self.name):
            print(f"Модель '{self.name}' не найдена, будет создана новая...")
            self.model = self.create_model(
                input_shape=input_shape,
                output_shape=output_shape,
                filters=filters,
                kernel_size=kernel_size,
                dense_size=dense_size,
            )
            # Также нужно удалить файл конфигурации
            if os.path.isfile(self.name + ".cfg"):
                os.remove(self.name + ".cfg")
        else:
            self.model = keras.models.load_model(
                self.name, custom_objects={"shifted_mse": shifted_mse}
            )
            self.Scaler = joblib.load(self.name + ".scaler")

        self.trained = True
        if not os.path.isfile(self.name + ".cfg"):
            self.datainfo = self.create_datainfo(
                input_shape=input_shape,
                output_shape=output_shape,
                predict_size=predict_size,
                x_std=0.001,
                y_std=0.001,
                timeunit=3600,
            )
        else:
            self.datainfo = DatasetInfo().load(self.name + ".cfg")

    def create_datainfo(
        self, input_shape, output_shape, predict_size, x_std, y_std, timeunit
    ):
        self.datainfo = DatasetInfo(
            input_shape=input_shape,
            output_shape=output_shape,
            future=predict_size,
            x_std=x_std,
            y_std=y_std,
            timeunit=timeunit,
        )
        self.datainfo.save(self.name + ".cfg")
        return self.datainfo

    def create_model(
        self, input_shape, output_shape, filters, kernel_size, dense_size,
    ):
        # self.model = multicol_conv1d_block(
        #     input_shape, output_shape, filters, kernel_size, dense_size,
        # )
        self.model = lstm_block(input_shape, output_shape, dense_size)
        return self.model

    def batch_generator(self, x_train, y_train, data_size, batch_size, sequence_length):
        while True:
            x_shape = (batch_size, sequence_length, 1)
            x_batch = np.zeroes(shape=x_shape, dtype=np.float)
            y_shape = (batch_size, sequence_length, 1)
            y_batch = np.zeros(shape=y_shape, dtype=np.float)
            for i in range(batch_size):
                idx = np.random.randint(data_size - sequence_length)
                x_batch[i] = x_train[idx : idx + sequence_length]
                y_batch[i] = y_train[idx : idx + sequence_length]
            yield (x_batch, y_batch)

    def get_input(self, prices, highs, lows):
        """
        Возвращает входной вектор для сети по массиву prices.
        """
        stride = 1
        shift = self.datainfo.future
        in_shape = self.datainfo.input_shape
        prices_diff = np.diff(np.array(prices))
        x = np.reshape(prices_diff, (-1, 1))
        x_scaled = np.reshape(self.Scaler.transform(x), (-1,))
        # data = tf.keras.preprocessing.timeseries_dataset_from_array(
        #     x_scaled, sequence_length=in_shape[0]
        # )
        # return data
        return roll(x_scaled, in_shape[0], stride)

    def load_dataset(
        self, tsv_file, count=0, skip=0, batch_size=256, validation_split=0.25
    ):
        """Подготовка обучающей выборки (x,y). Тип x и y - numpy.ndarray.
        Аргументы:
        csv_file - файл с ценами формата csv с колонками: 'date','time','open','high','low','close','tickvol','vol','spread'
        count - количество используемых примеров датасета
        skip - сдвиг относительно конца файла
        Возврат:
        x, y - размеченные данные типа numpy.array, сохранные на диск файл dataset_file"""
        # stride не менять, должно быть =1
        stride = 1  # шаг "нарезки" входных данных
        in_shape = self.datainfo.input_shape
        out_shape = self.datainfo.output_shape
        shift = self.datainfo.future
        in_size = self.datainfo._in_size()
        out_size = self.datainfo._out_size()  # out_shape[0]
        if not path.exists(tsv_file):
            print('Отсутствует файл "' + tsv_file + '"\nЗагрузка данных неуспешна')
            return None, None
        print("Чтение файла", tsv_file, "и создание обучающей выборки")
        data = pd.read_csv(
            tsv_file,
            sep="\t",
            header=0,
            dtype={
                "open": np.float64,
                "close": np.float64,
                "tickvol": np.float64,
                "vol": np.float64,
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
        if skip > len(data.index):
            print(f"Число skip больше числа строк данных: {skip}>{len(data.index)}")
            return None, None
        if count + skip > len(data.index):
            count = len(data.index) - skip

        times = data["time"]
        open_rates = data["open"]
        high_rates = data["high"]
        low_rates = data["low"]
        vol_rates = data["tickvol"]

        if skip == 0 and count == 0:
            left_bound = 0
            right_vound = len(times)
        elif skip == 0:
            left_bound = -count
            right_vound = len(times)
        elif count == 0:
            left_bound = 0
            right_bound = -skip
        else:
            left_bound = -count - skip
            right_bound = -skip

        times = np.array(data["time"])[left_bound:right_bound]

        opens = np.array(data["open"])[left_bound:right_bound]
        opens_diff = np.diff(opens)

        highs = np.array(data["high"])[left_bound:right_bound]
        highs_diff = np.diff(highs)

        lows = np.array(data["low"])[left_bound:right_bound]
        lows_diff = np.diff(lows)

        vols = np.array(data["tickvol"])[left_bound:right_bound]
        lows_diff = np.diff(vols)

        data_size = len(opens_diff)
        train_size = int((1 - validation_split) * data_size)

        # input_data = np.column_stack((opens_diff, highs_diff, lows_diff))
        x = opens_diff[: -in_shape[0] - shift]
        y = opens_diff[in_shape[0] + shift :]

        x_mean = x.mean()
        x_std = x.std()
        y_mean = y.mean()
        y_std = y.std()
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        self.Scaler.fit(x)
        x_scaled = self.Scaler.transform(x)
        y_scaled = self.Scaler.transform(y)
        joblib.dump(self.Scaler, self.name + ".scaler")
        x_scaled = np.reshape(x_scaled, (-1,))
        y_scaled = np.reshape(y_scaled, (-1,))
        # print(x_mean, x_std)

        # x_scaled = opens[: -in_shape[0] - shift]
        # y_scaled = opens[in_shape[0] + shift :]

        train_data = tf.keras.preprocessing.timeseries_dataset_from_array(
            x_scaled[:train_size],
            y_scaled[:train_size],
            sequence_length=in_shape[0],
            sequence_stride=stride,
            batch_size=batch_size,
            shuffle=True,
        )

        val_data = tf.keras.preprocessing.timeseries_dataset_from_array(
            x_scaled[train_size:],
            y_scaled[train_size:],
            sequence_length=in_shape[0],
            sequence_stride=stride,
            batch_size=batch_size,
        )

        self.datainfo.x_std = x_std
        self.datainfo.y_std = y_std
        self.datainfo.save(self.name + ".cfg")

        print(f"Загружено {data_size} примеров")
        return train_data, val_data

    def mass_center(self, x):
        shift = (len(x) - 1) / 2.0
        return np.sum(x * np.arange(len(x))) - shift
        return np.sum(x * np.arange(len(x))) - shift

    def predict(self, prices, highs, lows, verbose=1):
        """
        Вычисление результата для набора
        opens - массив размерности (n, input_size+1)
        """
        x = self.get_input(prices, highs, lows)
        if x is None:
            return None
        y = self.model.predict(x, use_multiprocessing=True, verbose=verbose)
        y = self.Scaler.inverse_transform(y)
        result = []
        for i in range(len(y)):
            price = float(y[i, 0])
            high = float(y[i, 0])
            low = float(y[i, 0])
            result.append((price, low, high, 0))
        return result

    def eval(self, opens):
        """
        opens - массив размерности (input_size)
        """
        if self.is_tflite():
            return self.tflite_predict([opens])[0]
        else:
            return self.predict([opens], verbose=0)[0]

    def shuffle_dataset(self, x, y):
        n = np.arange(len(x))
        np.random.shuffle(n)
        return x[n].astype("float64"), y[n].astype("float64")

    def train(self, dataset, val_datatset, batch_size, epochs):
        # x, y = self.shuffle_dataset(x, y)
        # Загрузим веса, если они есть
        ckpt = "ckpt/" + self.name + ".ckpt"
        try:
            self.model.load_weights(ckpt)
            print("Загружены веса последней контрольной точки " + self.name)
        except Exception as e:
            print(f"Не удалось загрузить веса контрольной точки: \n{e}\n")
            pass
        # Функция для tensorboard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_link = keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor="loss", patience=2 ** 8, min_delta=1e-4, restore_best_weights=True,
        )
        reduceLR = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=16, min_lr=0.000001
        )
        # backup = tf.keras.callbacks.ex.experimental.BackupAndRestore(backup_dir="backups/")
        backup = keras.callbacks.ModelCheckpoint(
            filepath=ckpt,
            monitor="val_loss",
            save_weights_only=True,
            save_best_only=True,
        )
        history = self.model.fit(
            dataset,
            validation_data=val_datatset,
            batch_size=batch_size,
            epochs=epochs,
            # validation_split=1.0 / 4.0,
            shuffle=True,
            use_multiprocessing=True,
            callbacks=[backup, early_stop, reduceLR, tensorboard_link],
        )
        self.model.save(self.name + ".h5")
        self.model.save(self.name)
        print("Модель " + self.name + " сохранена")
        return history


def train(modelname, datafile, input_shape, output_shape, future, batch_size, epochs):
    tf.profiler.experimental.stop()
    p = Predictor(
        modelname=modelname,
        input_shape=input_shape,
        output_shape=output_shape,
        predict_size=future,
        filters=2 ** 7,
        kernel_size=4,
        dense_size=2 ** 9,
    )
    # keras.utils.plot_model(p.model, show_shapes=True, to_file=modelname + ".png")
    dataset, val_datatset = p.load_dataset(
        tsv_file=datafile,
        batch_size=batch_size,
        count=8760 * 8,  # таймфреймы за x*N лет
        skip=1440,  # в часах
        validation_split=1 / 8,
    )
    if not dataset is None:
        history = p.train(dataset, val_datatset, batch_size=batch_size, epochs=epochs)
    else:
        print("Выполнение прервано из-за ошибки входных данных")


if __name__ == "__main__":
    batch_size = 2 ** 10
    for param in sys.argv:
        if param == "--gpu":
            batch_size = 2 ** 8
        elif param == "--cpu":
            batch_size = 2 ** 14
    train(
        modelname="models/59",
        datafile="datas/EURUSD_H1.csv",
        input_shape=(16, 1),
        output_shape=(16,),
        future=1,
        batch_size=batch_size,
        epochs=2 ** 14,
    )
# Debug
# Тест загрузки предиктора

