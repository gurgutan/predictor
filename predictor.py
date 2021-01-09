import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import datetime
from os import path
from tensorflow import keras
from tensorflow.keras.utils import plot_model


from dataloader import Dataloader
from models import (
    scored_boost,
    rbf_dense,
    spectral,
    dense_boost,
    trend_encoder,
    esum,
    dense_model,
    conv_model,
)
import sys


class Predictor(object):
    def __init__(
        self,
        datafile,
        model,
        input_width,
        label_width,
        shift=None,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=256,
    ):
        if shift == None:
            shift = label_width
        self.dataloader = Dataloader(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            ma=3,
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
        elif type(datafile) == pd.core.frame.DataFrame:
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
                "Ошибка загрузки модели модели. Параметр model должен быть либо строкой, либо моделью keras"
            )

    def __call__(self, data):
        # x = self.dataloader.make_input(data)
        return self.model(data[-self.dataloader.input_width - 1 :])

    def load_model(self, filename, lr=1e-5):
        # self.model = keras.models.load_model(self.name, custom_objects={"shifted_mse": shifted_mse})
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

    def plot_model(self):
        model_png_name = "models/" + self.model.name + ".png"
        keras.utils.plot_model(
            self.model,
            show_shapes=True,
            show_layer_names=False,
            to_file=model_png_name,
        )
        # self.model.summary()

    def fit(
        self,
        batch_size=256,
        epochs=8,
        use_tensorboard=True,
        use_early_stop=True,
        use_checkpoints=True,
        use_multiprocessing=True,
        verbose=1,
    ):
        log_dir = "logs/fit/" + start_fit_time.strftime("%Y_%m_%d-%H_%M_%S")
        start_fit_time = datetime.datetime.now()
        if use_checkpoints:
            ckpt = "ckpt/" + self.model.name + ".ckpt"
            backup = keras.callbacks.ModelCheckpoint(
                filepath=ckpt,
                monitor="loss",
                save_weights_only=True,
                save_best_only=True,
            )
            try:
                self.model.load_weights(ckpt)
                if verbose > 0:
                    print(
                        "Загружены веса последней контрольной точки " + self.model.name
                    )
            except Exception as e:
                pass

        callbacks = []
        if use_checkpoints:
            callbacks.append(backup)
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

    def evaluate(self):
        return self.model.evaluate(self.dataloader.test, verbose=1)

    def predict(self, data, verbose=0):
        """Вычисление результата для набора data - массив размерности n"""
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


if __name__ == "__main__":
    batch_size = 2 ** 10
    for param in sys.argv:
        if param == "--gpu":
            batch_size = 2 ** 12
        elif param == "--cpu":
            batch_size = 2 ** 16
        else:
            batch_size = 2 ** 12

    restarts_count = 2 ** 10
    dataset_segment = 1.0 / 8.0
    input_width = 2 ** 8
    label_width = 1
    columns = 2 ** 4

    # model = dense_boost(
    #     input_width,
    #     label_width,
    #     ensemble_size=columns,
    #     lr=1e-5,
    #     min_v=-3.0,
    #     max_v=3.0,
    #     name=f"dense-boost{ensemble_size}-{input_width}-{label_width}",
    # )
    model = scored_boost(
        input_width,
        label_width,
        prob_width=8,
        columns=8,
        lr=1e-6,
        min_v=-3.0,
        max_v=3.0,
        name=f"scored-boost{columns}-{input_width}-{label_width}",
    )

    predictor = Predictor(
        datafile="datas/EURUSD_H1 copy 3.csv",
        model=model,
        input_width=input_width,
        label_width=label_width,
        train_ratio=1.0 - 1.0 * dataset_segment,
        val_ratio=dataset_segment,
        test_ratio=0,
        batch_size=batch_size,
    )
    predictor.model.summary()
    for i in range(restarts_count):
        predictor.plot_model()
        print(f"\nМодель {model.name} проход №{i+1}/{restarts_count}\n")
        history = predictor.fit(batch_size=batch_size, epochs=2 ** 16)
        predictor.save_model()
        # perfomance = predictor.evaluate()
        print("Модель обновлена")

