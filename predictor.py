import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import random
from os import path
import time
import datetime
import sys
from patterns import conv2D, multiConv2D
from datainfo import DatasetInfo

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def roll(a, size, dx=1):
    shape = a.shape[:-1] + (int((a.shape[-1] - size) / dx) + 1, size)
    strides = a.strides + (a.strides[-1] * dx,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def embed(v, min_v, max_v, dim):
    """Возвращает бинарный вектор, длины dim"""
    step_size = (dim - 1) / (max_v - min_v)
    n = int(max(0, min(dim - 1, (v - min_v) * step_size)))
    result = np.zeros(dim, dtype="float32")
    # result = np.full(dim, -1, dtype="float32")
    result[n] = 1
    return result


def unembed(n, min_v, max_v, dim):
    step_size = (max_v - min_v) / (dim - 1)
    v = min_v + n * step_size
    return v


class Predictor(object):
    def __init__(
        self,
        modelname=None,
        input_shape=(16, 16, 1),
        output_shape=(64,),
        predict_size=16,
        filters=64,
        kernel_size=4,
        dense_size=8,
    ):
        self.trained = False
        if modelname == None:
            self.name = "default"
        else:
            self.name = path.splitext(modelname)[0]
        if not path.exists(self.name):
            self.model = self.create_model(
                input_shape=input_shape,
                output_shape=output_shape,
                filters=filters,
                kernel_size=kernel_size,
                dense_size=dense_size,
            )
        else:
            self.model = keras.models.load_model(self.name)
            self.trained = True
        if not os.path.isfile(self.name + ".cfg"):
            self.datainfo = self.create_datainfo(
                input_shape=input_shape,
                output_shape=output_shape,
                predict_size=predict_size,
                x_std=0.0004,
                y_std=0.0074,
            )
        else:
            self.datainfo = DatasetInfo().load(self.name + ".cfg")

    def create_datainfo(self, input_shape, output_shape, predict_size, x_std, y_std):
        self.datainfo = DatasetInfo(
            input_shape=input_shape,
            output_shape=output_shape,
            future=predict_size,
            x_std=x_std,
            y_std=y_std,
        )
        self.datainfo.save(self.name + ".cfg")
        return self.datainfo

    def create_model(
        self,
        input_shape,
        output_shape,
        conv_number=16,
        filters=64,
        kernel_size=2,
        dense_size=8,
    ):
        self.model = conv2D(
            input_shape=input_shape,
            output_shape=output_shape,
            filters=filters,
            kernel_size=kernel_size,
            dense_size=dense_size,
        )
        return self.model

    def get_input(self, close_list):
        """
        Возвращает входной вектор для сети по массиву close_list.
        Размерность close_list: (любая, input_size+1)
        """
        result = []
        in_size = self.datainfo._in_size()
        for c in close_list:
            if len(c) < in_size + 1:
                print(
                    "Ошибка размерности input_data: "
                    + str(len(c))
                    + "<"
                    + str(in_size + 1)
                )
                x = np.zeros(
                    self.datainfo.input_shape[0],
                    self.datainfo.input_shape[1],
                    self.datainfo.input_shape[2],
                )
            else:
                closes = np.array(c[-in_size - 1 :])
                d = np.nan_to_num(
                    np.diff(closes) / self.datainfo.x_std,
                    posinf=self.datainfo._x_inf(),
                    neginf=-self.datainfo._x_inf(),
                )
                x = np.reshape(
                    d,
                    (
                        self.datainfo.input_shape[0],
                        self.datainfo.input_shape[1],
                        self.datainfo.input_shape[2],
                    ),
                )
            result.append(x)
        if len(result) == 0:
            return None
        return np.stack(result, axis=0)

    def load_dataset(self, csv_file, count=0, skip=0):
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
        future = self.datainfo.future
        in_size = self.datainfo._in_size()
        out_size = self.datainfo._out_size()  # out_shape[0]
        if not path.exists(csv_file):
            print('Отсутствует файл "' + csv_file + '"\nЗагрузка данных неуспешна')
            return None, None
        print("Чтение файла", csv_file, "и создание обучающей выборки")
        data = pd.read_csv(
            csv_file,
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
        if skip == 0 and count == 0:
            closes = data["close"]
        elif skip == 0:
            closes = data["close"][-count:]
        elif count == 0:
            closes = data["close"][:-skip]
        else:
            closes = data["close"][-count - skip : -skip]
        # получим серию с разницами цен закрытия
        d = np.diff(np.array(closes), n=1)
        # нормируем серию стандартным отклонением
        x_std = d.std()
        # изменим datainfo
        self.datainfo.x_std = float(x_std)
        d = np.nan_to_num(
            d / x_std, posinf=self.datainfo._x_inf(), neginf=-self.datainfo._x_inf()
        )
        x_data = roll(d[:-future], in_size, stride)
        x_forward = roll(d[in_size:], future, stride)
        y_data = np.sum(x_forward * self.datainfo.x_std, axis=1)
        self.datainfo.y_std = float(y_data.std())
        x = np.reshape(x_data, (x_data.shape[0], in_shape[0], in_shape[1], in_shape[2]))
        y = np.zeros((y_data.shape[0], out_shape[0]))
        for i in range(y.shape[0]):
            y[i] = embed(
                y_data[i], self.datainfo._y_min(), self.datainfo._y_max(), out_size,
            )
        self.datainfo.save(self.name + ".cfg")
        return x.astype("float32"), y.astype("float32")

    def train(self, x, y, batch_size, epochs):
        # Загрузим веса, если они есть
        ckpt = "ckpt/" + self.name + ".ckpt"
        try:
            self.model.load_weights(ckpt)
            print("Загружены веса последней контрольной точки " + self.name)
        except Exception as e:
            pass
        # Функция для tensorboard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_link = keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_mean_absolute_error",
            patience=16,
            min_delta=1e-3,
            restore_best_weights=True,
        )
        cp_save = keras.callbacks.ModelCheckpoint(filepath=ckpt, save_weights_only=True)
        history = self.model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            shuffle=True,
            use_multiprocessing=True,
            callbacks=[early_stop, cp_save, tensorboard_link],
        )
        self.model.save(self.name)
        print("Модель " + self.name + " сохранена")
        return history

    def predict(self, closes, verbose=1):
        """
        Вычисление результата для набора
        closes - массив размерности (n, input_size+1)
        """
        x = self.get_input(closes)
        if x is None:
            return None
        y = self.model.predict(x, use_multiprocessing=True, verbose=verbose)
        n = np.argmax(y, axis=1)
        y_n = y[np.arange(len(y)), n]
        result = []
        for i in range(len(y_n)):
            low = (
                unembed(
                    n[i],
                    self.datainfo._y_min() / self.datainfo.y_std,
                    self.datainfo._y_max() / self.datainfo.y_std,
                    self.datainfo._out_size(),
                )
                * self.datainfo.y_std
            )
            high = (
                unembed(
                    n[i] + 1,
                    self.datainfo._y_min() / self.datainfo.y_std,
                    self.datainfo._y_max() / self.datainfo.y_std,
                    self.datainfo._out_size(),
                )
                * self.datainfo.y_std
            )
            result.append((low, high, float(y_n[i])))
        return result

    def eval(self, closes):
        """
        closes - массив размерности (input_size)
        """
        return self.predict([closes], verbose=0)


def train(modelname, batch_size, epochs):
    input_shape = (16, 16, 1)
    output_shape = (8,)
    predict_size = 16
    p = Predictor(
        modelname=modelname,
        #   datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5',
        input_shape=input_shape,
        output_shape=output_shape,
        predict_size=predict_size,
        filters=16,
        kernel_size=2,
        dense_size=64,
    )
    x, y = p.load_dataset(
        csv_file="datas/EURUSD_M5_200001030000_202006122350.csv", count=2 ** 19
    )
    keras.utils.plot_model(p.model, show_shapes=True)
    if not x is None:
        history = p.train(x, y, batch_size=batch_size, epochs=epochs)
    else:
        print("Выполнение прервано из-за ошибки входных данных")


if __name__ == "__main__":
    for param in sys.argv:
        if param == "--train":
            train("models/19", batch_size=2 ** 9, epochs=2 ** 10)
# Debug
# Тест загрузки предиктора
