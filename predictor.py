import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import math
import tensorflow as tf
import datetime
from os import path
from tensorflow import keras
from tensorflow.keras.utils import plot_model

from dataloader import Dataloader
from models import *
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Predictor(object):
    def __init__(
        self,
        datafile,
        model,
        input_width,
        label_width,
        sample_width=1,
        shift=1,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=256,
    ):
        self.dataloader = Dataloader(
            input_width=input_width,
            label_width=label_width,
            sample_width=sample_width,
            shift=shift,
            batch_size=batch_size,
        )
        if not datafile is None:
            self.dataloader.load(
                datafile,
                input_column="open",
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
        if type(model) == str:
            self.load_model(model)
        elif type(model) == tf.python.keras.engine.functional.Functional:
            self.model = model
        else:
            print(
                "Ошибка загрузки модели модели. Параметр model должен быть либо строкой, либо моделью keras"
            )

    def load_model(self, filename):
        # self.model = keras.models.load_model(self.name, custom_objects={"shifted_mse": shifted_mse})
        self.model = keras.models.load_model("models/" + self.name)

    def print_model(self):
        model_png_name = "models/" + self.model.name + ".png"
        keras.utils.plot_model(self.model, show_shapes=True, to_file=model_png_name)
        self.model.summary()

    def fit(self, batch_size=256, epochs=8):
        start_fit_time = datetime.datetime.now()
        ckpt = "ckpt/" + self.model.name + ".ckpt"
        try:
            self.model.load_weights(ckpt)
            print("Загружены веса последней контрольной точки " + self.model.name)
        except Exception as e:
            pass
        log_dir = "logs/fit/" + start_fit_time.strftime("%Y_%m_%d-%H_%M_%S")
        tensorboard_link = keras.callbacks.TensorBoard(
            log_dir=log_dir, write_graph=True
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2 ** 10,
            min_delta=1e-5,
            restore_best_weights=True,
        )
        backup = keras.callbacks.ModelCheckpoint(
            filepath=ckpt,
            monitor="val_loss",
            save_weights_only=True,
            save_best_only=True,
        )
        history = self.model.fit(
            self.dataloader.train,
            validation_data=self.dataloader.val,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            use_multiprocessing=True,
            callbacks=[backup, early_stop, tensorboard_link],
        )
        end_fit_time = datetime.datetime.now()
        self.print_model()
        self.model.save("models/" + self.model.name + ".h5")
        self.model.save("models/" + self.model.name)
        print(
            f"Начало: {start_fit_time} конец: {end_fit_time} время: {end_fit_time-start_fit_time}"
        )
        return history

    def evaluate(self):
        return self.model.evaluate(self.dataloader.test, verbose=1)

    def predict(self, data, verbose=1):
        """Вычисление результата для набора data - массив размерности n"""
        x = self.dataloader.make_input(data)
        y = self.model.predict(x, use_multiprocessing=True, verbose=verbose)
        result = self.dataloader.make_output(y)
        return result

    def iterate(self, data, steps=4):
        results = []
        data = data[-self.dataloader.input_width :]
        inputs = self.dataloader.make_input(data)
        for i in range(steps):
            outputs = self.model(inputs)
            data = np.append(data, float(outputs[0]))[-self.dataloader.input_width :]
            inputs = self.dataloader.make_input(data)
            results += [float(outputs[0])]
        return results


if __name__ == "__main__":
    batch_size = 2 ** 10
    for param in sys.argv:
        if param == "--gpu":
            batch_size = 2 ** 8
        elif param == "--cpu":
            batch_size = 2 ** 15
        else:
            batch_size = 2 ** 12

    sample_width = 1
    input_width = 32
    sections = int(math.log2(input_width))
    model = trend_encoder((input_width, sample_width), units=2 ** 9, sections=sections)
    # model = lstm_block((input_width, sample_width), units=2 ** 9, count=2)
    # model = spectral((64, 1), 256, width=6, depth=4)
    # model =   multi_dense((64, sample_width), 256, 16)
    predictor = Predictor(
        datafile="datas/EURUSD_H1 copy.csv",
        model=model,
        input_width=input_width,
        label_width=1,
        sample_width=sample_width,
        shift=1,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=batch_size,
    )
    history = predictor.fit(batch_size=batch_size, epochs=2 ** 16)
    perfomance = predictor.evaluate()

