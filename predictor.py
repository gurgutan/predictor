import pydot
import sys
from models import *
from dataloader import Dataloader
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from os import path
import datetime
import tensorflow as tf
import math
import pandas as pd
import numpy as np
import os

# import shutil
from subprocess import call


# os.environ["ENV TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# from tensorflow.keras.utils import plot_model


class Predictor(object):
    def __init__(
        self,
        datafile,
        model,
        input_width,
        label_width,
        shift=None,
        clip_by_value=4.0,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=256,
    ):
        if shift is None:
            shift = label_width
        self.dataloader = Dataloader(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            clip_by_value=clip_by_value,
            batch_size=batch_size,
        )
        if type(datafile) == str:
            self.dataloader.load_tsv(
                datafile,
                input_column="open",
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
        elif type(datafile) == pd.core.frame.DataFrame:  # type: ignore
            self.dataloader.load_df(
                datafile,
                input_column="open",
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
        if type(model) == str:
            self.load_model(model)
        elif isinstance(model, tf.keras.Model):
            self.model = model
        else:
            print(
                "Ошибка загрузки модели модели. \n"
                + "Параметр model должен быть либо строкой, либо моделью keras"
            )

    def __call__(self, data):
        # x = self.dataloader.make_input(data)
        return self.model(data[-self.dataloader.input_width - 1 :])

    def load_model(self, filename, lr=1e-5):
        # self.model = keras.models.load_model(self.name, custom_objects={
        #                                      "shifted_mse": shifted_mse})
        self.model = keras.models.load_model(filename, compile=False)
        self.model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        return self.model

    def save_model(self):
        # self.model.save("models/" + self.model.name + ".h5")
        self.model.save("models/" + self.model.name)

    def plot(self):
        model_png_name = "models/" + self.model.name + ".png"
        plot_model(
            self.model,
            show_shapes=True,
            show_layer_names=False,
            rankdir="LR",
            to_file=model_png_name,
        )

    def fit(
        self,
        batch_size=256,
        epochs=2,
        use_tensorboard=True,
        use_early_stop=True,
        use_checkpoints=True,
        use_multiprocessing=True,
        verbose=1,
    ):
        callbacks = []
        start_fit_time = datetime.datetime.now()
        log_dir = "logs/fit/" + start_fit_time.strftime("%Y_%m_%d-%H_%M_%S")
        if use_checkpoints:
            ckpt_filename = "ckpt/" + self.model.name + ".ckpt"
            backup = keras.callbacks.ModelCheckpoint(
                filepath=ckpt_filename,
                monitor="loss",
                save_weights_only=True,
                save_best_only=True,
                save_freq=2 ** 9,
            )
            callbacks.append(backup)
            try:
                self.model.load_weights(ckpt_filename)
                if verbose > 0:
                    print(
                        "Загружены веса последней контрольной точки " + self.model.name
                    )
            except Exception as e:
                pass

        if use_tensorboard:
            tb = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
            callbacks.append(tb)
        if use_early_stop:
            es = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2 ** 4,
                min_delta=1e-5,
                restore_best_weights=True,
            )
            callbacks.append(es)

        history = self.model.fit(
            self.dataloader.train,
            validation_data=self.dataloader.val,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            use_multiprocessing=use_multiprocessing,
            verbose=verbose,
            callbacks=callbacks,
        )
        end_fit_time = datetime.datetime.now()
        delta_time = end_fit_time - start_fit_time
        if verbose > 0:
            print(f"\nВремя {start_fit_time}->{end_fit_time} : {delta_time}")
        return history

    def evaluate(self, x):
        return self.model(x, training=False)

    def predict(self, data, verbose=0):
        """
        Вычисление результата для набора data - массив размерности n
        Результат - массив numpy размерности (n-in_size, out_size)
        """
        x = self.dataloader.make_input(data)
        f = self.model.predict(x, use_multiprocessing=True, verbose=verbose)
        y = self.dataloader.make_output(f)
        # result = self.dataloader.make_output(y)
        return y

    def iterate(self, inputs, steps=1):
        results = []
        # input_width + 1 нужно для вычисления np.diff
        size = self.dataloader.input_width + 1
        for i in range(0, steps):
            inputs = inputs[-size:]
            output = float(self.predict(inputs, verbose=0)[-1][0])
            inputs = np.append(inputs, inputs[-1] + output)
            results.append(output)
        return results

    def copy_model(self, dst_folder):
        src = "models/" + self.model.name
        call(["cp", "-arvu", src, dst_folder])
        # shutil.16copy(src, dst_folder)


# ===========================================================================
# Точка входа
# ===========================================================================
if __name__ == "__main__":
    batch_size = 2 ** 15 + 2 ** 12
    for param in sys.argv:
        if param == "--gpu":
            batch_size = 2 ** 14 + 2 ** 13
        elif param == "--cpu":
            batch_size = 2 ** 15

    dataset_segment = 1.0 / 8.0
    dataset_rows = 2 ** 16
    input_width = 32
    label_width = 1
    columns = 32
    learning_rate = 1e-3
    is_training = False
    clip_value = 2.0
    dropout = 1.0 / 2.0

    model = t1(
        input_width,
        label_width,
        columns=columns,
        lr=learning_rate,
        min_v=-clip_value,
        max_v=clip_value,
        training=is_training,
        dropout=dropout,
    )

    data_file = "datas/EURUSD_H1.csv"
    predictor = Predictor(
        datafile=data_file,
        model=model,
        input_width=input_width,
        label_width=label_width,
        clip_by_value=clip_value,
        train_ratio=1.0 - 1.0 * dataset_segment,
        val_ratio=dataset_segment,
        test_ratio=0,
        batch_size=batch_size,
    )

    # predictor.plot()
    predictor.model.summary()
    onednn_enabled = int(os.environ.get("TF_ENABLE_ONEDNN_OPTS", "0"))
    # print("MKL enabled :", onednn_enabled)
    print("\nTensorflow version", tf.__version__)

    i = 0
    while True:
        i += 1
        print(f"\nМодель {model.name} проход №{i}\n")
        predictor.dataloader.load_tsv(
            data_file,
            input_column="open",
            train_ratio=1.0 - 1.0 * dataset_segment,
            val_ratio=dataset_segment,
            test_ratio=0,
            nrows=dataset_rows,
        )
        history = predictor.fit(
            use_tensorboard=False,
            use_early_stop=False,
            use_checkpoints=True,
            batch_size=batch_size,
            epochs=2 ** 11,
        )

        predictor.save_model()
        # predictor.copy_model(
        # "smb://keenetic-smb.local/temp/dev/predictor.rc/models/")
        # perfomance = predictor.evaluate()
        print("Модель обновлена")
