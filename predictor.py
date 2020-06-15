import io
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import random
from os import path
import os
import time
import datetime
import sys

import models
from datainfo import DatasetInfo


def roll(a, size, dx=1):
    shape = a.shape[:-1] + (int((a.shape[-1] - size) / dx) + 1, size)
    strides = a.strides + (a.strides[-1] * dx, )
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def embed(v, min_v, max_v, dim):
    """Возвращает бинарный вектор, длины dim"""
    step_size = (dim - 1) / (max_v - min_v)
    v = min(max_v, v)
    v = max(min_v, v)
    n = int((v - min_v) * step_size)
    result = np.zeros(dim, dtype='float32')
    # result = np.full(dim, -1, dtype='float32')
    result[n] = 1
    return result


class Predictor(object):
    def __init__(self,
                 filename='',
                 input_shape=(16, 16, 1),
                 output_shape=(64, ),
                 predict_size=16,
                 filters=64,
                 kernel_size=4,
                 dense_size=8):
        if (filename == ''):
            self.name = 'default'
        else:
            self.name = path.splitext(filename)[0]
        if (not os.path.isfile(self.name + '.h5')):
            self.model = self.create_model(input_shape=input_shape,
                                           output_shape=output_shape,
                                           filters=filters,
                                           kernel_size=kernel_size,
                                           dense_size=dense_size)
        else:
            self.model = keras.models.load_model(self.name + '.h5')
        if (not os.path.isfile(self.name + '.cfg')):
            self.datainfo = self.create_datainfo(input_shape=input_shape,
                                                 output_shape=output_shape,
                                                 predict_size=predict_size)
        else:
            self.datainfo = DatasetInfo().load(self.name + '.cfg')

    def create_datainfo(self,
                        input_shape=(16, 16, 1),
                        output_shape=(16, 1),
                        predict_size=16):
        self.datainfo = DatasetInfo(input_shape=input_shape,
                                    output_shape=output_shape,
                                    predict_size=predict_size,
                                    x_std=0.0004,
                                    y_std=0.0004 * 16,
                                    time_unit=300)
        self.datainfo.save(self.name + '.cfg')
        return self.datainfo

    def create_model(self,
                     input_shape,
                     output_shape,
                     conv_number=16,
                     filters=64,
                     kernel_size=4,
                     dense_size=8):
        self.model = models.conv2D(input_shape=input_shape,
                                   output_shape=output_shape,
                                   filters=filters,
                                   kernel_size=kernel_size,
                                   dense_size=dense_size)
        return self.model

    def get_single_input(self, close_list):
        """Возвращает входной вектор для сети по списку цен x_data.
        Размер x_data должен быть не менее чем input_size+1.
        Если len(x_data) больше, то беруться последние input_size+1 значений"""
        input_size = self.datainfo.input_shape[0] * self.datainfo.input_shape[
            1] * self.datainfo.input_shape[2]
        if (len(close_list) < input_size + 1):
            print("Ошибка размерности input_data: " + str(len(close_list)) +
                  "<" + str(input_size + 1))
            return np.zeros(
                (1, self.datainfo.input_shape[0], self.datainfo.input_shape[1],
                 self.datainfo.input_shape[2]))
        infinity = self.datainfo.predict_size * 2
        x = np.array(close_list[-input_size - 1:])
        std = self.datainfo.x_std
        d = np.nan_to_num(np.diff(x) / std, posinf=infinity, neginf=-infinity)
        x = np.reshape(
            d, (1, self.datainfo.input_shape[0], self.datainfo.input_shape[1],
                self.datainfo.input_shape[2]))
        return x

    def load_dataset(self,
                     csv_file='./data/stocks.csv',
                     dataset_file='./data/dataset.npz',
                     count=0,
                     skip=0):
        """Подготовка обучающей выборки (x,y). Тип x и y - numpy.ndarray.
        Аргументы:
        csv_file - файл с ценами формата csv с колонками: 'date','time','open','high','low','close','tickvol','vol','spread'
        dataset_file - имя файла для сохранения входных размеченных данных [x,y]
        count - количество примеров датасета
        skip - сдвиг относительно конца файла
        input_shape - форма входных данных сети. Должна иметь 3 оси
        output_shape - форма выходных данных сети. Должна иметь 1 ось
        Возврат:
        x, y - размеченные данные типа numpy.array, сохранные на диск файл dataset_file"""

        # TODO Переделать под работу только с csv
        # Т.к. выход - нормирован стандартным отклонением, максимальное и минимальное значение примем равным predict
        in_shape = self.datainfo.input_shape
        out_shape = self.datainfo.output_shape
        predict_size = self.datainfo.predict_size
        if (path.isfile(dataset_file)):
            npzfile = np.load(dataset_file)
            x = npzfile['arr_0']
            y = npzfile['arr_1']
            if (tuple(list(x.shape)[1:]) != in_shape
                    or tuple(list(y.shape)[1:]) != out_shape):
                print(
                    'Формат входных и/или выходных данных не совпадает с размерностью модели. Данные будут размечены заново'
                )
                os.remove(dataset_file)
            else:
                self.datainfo.y_std = float(np.std(y))
                self.datainfo.save(self.name + '.cfg')
                return x.astype("float32"), y.astype("float32")

        if (not path.isfile(csv_file)):
            print('Отсутствует файл "' + csv_file +
                  '". Разметка данных не выполнена')
            print('Загрузка данных неуспешна.')
            return None, None
        print('Чтение файла', csv_file, 'и создание обучающей выборки')
        data = pd.read_csv(csv_file,
                           sep='\t',
                           dtype={
                               'open': np.float32,
                               'close': np.float32,
                               'tickvol': np.float32,
                               'vol': np.float32
                           },
                           names=[
                               'date', 'time', 'open', 'high', 'low', 'close',
                               'tickvol', 'vol', 'spread'
                           ])
        input_size = in_shape[0] * in_shape[1] * in_shape[2]
        output_size = out_shape[0]
        stride = 2
        if (skip == 0 or count == 0):
            closes = data['close']
        else:
            closes = data['close'][-count - skip:-skip]
        # получим серию с разницами цен закрытия
        d = (closes[1:].reset_index(drop=True) -
             closes[:-1].reset_index(drop=True)).fillna(0)
        # нормируем серию стандартным отклонением
        x_std = d.std()
        infinity = x_std * 4
        d = np.nan_to_num(d.div(x_std), posinf=infinity, neginf=-infinity)
        x_data = roll(d[:-predict_size], input_size, stride)
        x_forward = roll(d[input_size:], predict_size, stride)
        y_data = np.sum(x_forward, axis=1)
        y_std = y_data.std()
        # в качестве рассматриваемого диапазона берем [-3*std, 3*std]
        min_v = -y_std * 3
        max_v = y_std * 3
        x = np.reshape(
            x_data, (x_data.shape[0], in_shape[0], in_shape[1], in_shape[2]))
        # y = np.full((y_data.shape[0], out_shape[0]), 0.0, dtype='float32')
        y = np.zeros((y_data.shape[0], out_shape[0]), dtype='float32')
        for i in range(y.shape[0]):
            y[i] = embed(y_data[i], min_v, max_v, out_shape[0])
        np.savez(dataset_file, x, y)
        # изменим datainfo
        self.datainfo.x_std = float(x_std)
        self.datainfo.y_std = float(y_std)
        self.datainfo.save(self.name + '.cfg')
        return x.astype("float32"), y.astype("float32")

    def train(self, x, y, batch_size=64, epochs=1024):
        # Загрузим веса, если они есть
        ckpt = 'ckpt/' + self.name + '.ckpt'
        try:
            self.model.load_weights(ckpt)
            print('Загружены веса последней контрольной точки ' + self.name)
        except Exception as e:
            pass
        # Функция для tensorboard
        log_dir = "./logs/fit/" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")
        tensorboard_link = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                       histogram_freq=1,
                                                       write_graph=True)
        early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                                   patience=16,
                                                   min_delta=1e-4)
        cp_save = keras.callbacks.ModelCheckpoint(filepath=ckpt,
                                                  save_weights_only=True)
        history = self.model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            use_multiprocessing=True,
            callbacks=[early_stop, cp_save, tensorboard_link])
        self.model.save(self.name)
        print('Модель ' + self.name + ' сохранена')
        return history

    def infer(self, close_list):
        """
        Возвращает число float32 с прогнозом цены
        Параметры:
        close_list - список текущих цен закрытия
        """
        x = self.get_single_input(close_list)
        if (not x.any()):
            return None
        # возвращаем денормализованный результат
        y = self.model(x, training=False)[0].numpy()
        y_str = ''
        for v in y:
            if (y_str != ''):
                y_str += ' '
            y_str += str(v.round(6))
        return y_str


def train(modelname, dataset_file, batch_size=2**8, epochs=2**2):
    input_shape = (16, 16, 1)
    output_shape = (17, )
    predict_size = 16
    count = 2**20
    p = Predictor(
        filename=modelname,
        #   datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5',
        input_shape=input_shape,
        output_shape=output_shape,
        predict_size=predict_size,
        filters=64,
        kernel_size=4,
        dense_size=64)
    x, y = p.load_dataset(dataset_file=dataset_file, count=count, skip=2**15)
    if (x.any() and y.any()):
        history = p.train(x, y, batch_size=2**8, epochs=4**6)
    else:
        print('Выполнение  прервано из-за ошибки входных данных')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

for param in sys.argv:
    if (param == '--train'):
        train('results/4.h5', 'data/dataset.npz', batch_size=2**9, epochs=4)
# Debug
# Тест загрузки предиктора

# x = [1.10 + random.random() / 100.0 for i in range(2048)]
# print(p.infer(x))
