import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow.keras.layers
from functools import reduce
import pywt
import operator
from rbflayer import RBFLayerExp
from tensorflow.python.keras.layers.recurrent import LSTM

# tf.compat.v1.disable_eager_execution()


def reduce_mul(t: tuple) -> int:
    return reduce(operator.mul, self.output_shape)


def shifted_mse(y_true, y_pred):
    # d = tf.keras.losses.cosine_similarity(y_true, y_pred)
    mse = tf.keras.losses.mean_squared_error
    mae = keras.losses.mean_absolute_error
    y_true_sign = tf.math.softsign(y_true)
    y_pred_sign = tf.math.softsign(y_pred)
    d = mse(y_true, y_pred) * mae(y_true_sign, y_pred_sign)
    return d  # tf.reduce_mean(d, axis=-1)


def wave_len(input_shape, wavelet, mode):
    w = wavelet
    filter_len = w.dec_len
    mode = "zero"
    return pywt.dwt_coeff_len(input_shape[0], filter_len, mode=mode)


def lstm_block(input_shape, output_shape, units, count=2):
    inputs = keras.Input(shape=(input_shape[0], 1), name="inputs")
    x = inputs
    # x = layers.LayerNormalization(axis=1)(x)
    branch1 = branch2 = x
    for i in range(count - 1):
        branch1 = layers.LSTM(units, return_sequences=True)(branch1)
        branch2 = layers.Conv1D(units, 4, padding="same", activation="relu")(branch2)

    x = layers.Multiply()([branch1, branch2])

    # x = layers.GRU(units, return_sequences=False)(x)

    # forward_layer = LSTM(units, return_sequences=True)
    # backward_layer = LSTM(units, return_sequences=True, go_backwards=True)
    # x = layers.Bidirectional(forward_layer, backward_layer=backward_layer)(x)
    # x = layers.Dropout(1 / 16)(x)

    # x = layers.Dense(64, activation="softsign",)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    outputs = layers.Activation("linear")(x)
    model = keras.Model(inputs, outputs)

    MAE = keras.metrics.MeanAbsoluteError()
    MSE = keras.metrics.MeanSquaredError()
    model.compile(
        loss=shifted_mse,
        # loss=keras.losses.MeanSquaredError(),
        # loss=keras.losses.MeanAbsoluteError(),
        # loss=keras.losses.CosineSimilarity(),
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        metrics=[MAE],
    )
    print(model.summary())
    return model


def dense_block(input_shape, output_shape, units, count=3):
    l2_reg = keras.regularizers.l2(l=1e-6)
    wavelet_len = wave_len(input_shape, pywt.Wavelet("rbio3.1"), mode="zero")
    inputs = keras.Input(shape=(wavelet_len,), name="inputs")
    x = inputs
    x = layers.BatchNormalization()(x)
    # x = RBFLayerExp(units)(x)
    for i in range(count):
        x = layers.Dense(
            units,
            activation="softsign",
            bias_regularizer=l2_reg,
            kernel_regularizer=l2_reg,
        )(x)
        x = layers.BatchNormalization()(x)

        units = units * 2

    x = layers.Dropout(1.0 / 2.0)(x)
    outputs = layers.Dense(
        output_shape[0],
        activation="softmax",
        bias_regularizer=l2_reg,
        kernel_regularizer=l2_reg,
        name="outputs",
    )(x)

    model = keras.Model(inputs, outputs)
    AUC = keras.metrics.AUC()
    MAE = keras.metrics.MeanAbsoluteError()
    model.compile(
        # loss=keras.losses.CategoricalCrossentropy(),
        loss=keras.losses.CosineSimilarity(),
        # loss=keras.losses.KLDivergence(),
        # loss=keras.losses.Hinge(),
        # loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        metrics=[MAE],
    )
    print(model.summary())
    return model


def rnn_block(n, inputs):
    x = inputs
    for i in range(n):
        x = layers.SimpleRNN(x.shape[-2])(x)
        x = layers.Reshape((x.shape[-1], 1))(x)
    return x


def loc1d(input_shape, output_shape, filters, kernel_size, dense_size):
    l2_reg = keras.regularizers.l2(l=1e-7)
    rnd_uniform = keras.initializers.RandomUniform()
    wavelet_len = pywt.swt_max_level(input_shape[0]) * input_shape[0]
    inputs = keras.Input(shape=(wavelet_len, 1), name="inputs")
    # x = inputs[:, :, : input_shape[0] // 2]
    x = inputs
    x = layers.LayerNormalization(axis=[1, 2])(x)
    # x = layers.BatchNormalization()(x)
    ksize = kernel_size
    f = filters
    i = 0
    while ksize > 1 and i < 64:
        x = layers.LocallyConnected1D(
            f,
            ksize,
            activation="elu",
            bias_regularizer=l2_reg,
            kernel_regularizer=l2_reg,
            bias_initializer=rnd_uniform,
            kernel_initializer=rnd_uniform,
        )(x)
        ksize = min(x.shape.as_list()[1:] + [ksize])
        if ksize > 1:
            x = layers.MaxPool1D(pool_size=2)(x)
        ksize = min(x.shape.as_list()[1:] + [ksize])
        f *= 2
        i += 1

    x = layers.Flatten()(x)
    x = layers.Dropout(1.0 / 2.0)(x)
    x = layers.Dense(
        dense_size,
        activation="relu",
        bias_initializer=rnd_uniform,
        bias_regularizer=l2_reg,
        kernel_initializer=rnd_uniform,
        kernel_regularizer=l2_reg,
    )(x)
    outputs = layers.Dense(
        output_shape[0],
        activation="softmax",
        bias_initializer=rnd_uniform,
        bias_regularizer=l2_reg,
        kernel_initializer=rnd_uniform,
        kernel_regularizer=l2_reg,
        name="outputs",
    )(x)

    model = keras.Model(inputs, outputs)

    ROC = keras.metrics.AUC()
    model.compile(
        loss=keras.losses.CosineSimilarity(),
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        metrics=[ROC],
    )
    print(model.summary())
    return model


def sep_conv1d(f, ksize, padding="same"):
    l2_reg = keras.regularizers.l2(l=1e-7)
    rnd_uniform = keras.initializers.RandomUniform()
    return layers.SeparableConv1D(
        f,
        ksize,
        padding=padding,
        activation="relu",
        bias_regularizer=l2_reg,
        kernel_regularizer=l2_reg,
        bias_initializer=rnd_uniform,
        kernel_initializer=rnd_uniform,
    )


def sepconv1d_block(x, f, ksize, padding="same", count=8, slices=8):
    z = []
    for j in range(slices):
        z += [sep_conv1d(f, ksize, padding)(x)]
    for i in range(count):
        for j in range(slices):
            z[j] = sep_conv1d(f, ksize, padding)(z[j])
    x = layers.Add()(z)
    return x


def multicol_conv1d_block(
    input_shape, output_shape, filters, kernel_size, dense_size, rows=4
):
    f_max = 2 ** 9
    l2_reg = keras.regularizers.l2(l=1e-7)
    rnd_uniform = keras.initializers.RandomUniform()
    w = pywt.Wavelet("db8")
    filter_len = w.dec_len
    mode = "zero"
    level = 4
    wavelet_len = pywt.dwt_coeff_len(input_shape[0], filter_len, mode=mode)
    width = wavelet_len
    columns = 4
    inputs = keras.Input(shape=(wavelet_len, 1), name="inputs")
    f = filters
    x = inputs
    z = []
    for c in range(columns):
        z += [layers.LocallyConnected1D(1, 1, 1)(x)]
    for r in range(rows):
        for c in range(columns):
            z[c] = layers.SeparableConv1D(
                min(f, f_max), kernel_size, padding="valid", name=f"sc{r}-{c}"
            )(z[c])
            # z[c] = layers.BatchNormalization()(z[c])
            z[c] = layers.Activation("relu")(z[c])
            # z[c] = layers.MaxPooling1D(pool_size=2, name=f"mp{r}-{c}")(z[c])
        f *= 2
    x = layers.Add()(z)
    # x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(pool_size=8)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(1.0 / 2.0)(x)
    x = layers.Dense(
        dense_size,
        activation="relu",
        bias_initializer=rnd_uniform,
        bias_regularizer=l2_reg,
        kernel_initializer=rnd_uniform,
        kernel_regularizer=l2_reg,
    )(x)
    outputs = layers.Dense(
        output_shape[0],
        activation="softmax",
        bias_initializer=rnd_uniform,
        bias_regularizer=l2_reg,
        kernel_initializer=rnd_uniform,
        kernel_regularizer=l2_reg,
        name="outputs",
    )(x)
    model = keras.Model(inputs, outputs)
    ROC = keras.metrics.AUC()
    model.compile(
        # loss=keras.losses.CategoricalCrossentropy(),
        loss=keras.losses.CosineSimilarity(),
        # loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        metrics=[ROC],
    )
    print(model.summary())
    return model


def reslike(input_shape, output_shape, filters, kernel_size, dense_size):
    l2_reg = keras.regularizers.l2(l=1e-7)
    rnd_uniform = keras.initializers.RandomUniform()
    w = pywt.Wavelet("db8")
    filter_len = w.dec_len
    mode = "zero"
    level = 4
    wavelet_len = pywt.dwt_coeff_len(input_shape[0], filter_len, mode=mode)
    inputs = keras.Input(shape=(wavelet_len, 2), name="inputs")
    x = inputs
    # x = layers.LayerNormalization(axis=[1, 2])(x)
    ksize = kernel_size
    f = filters
    i = 0
    while ksize > 1 and i < 8:
        x = sep_conv1d(f, ksize)(x)
        residual = x
        x = sepconv1d_block(x, f, ksize, "same", 4, 2)
        x = layers.Add()([x, residual])
        x = layers.BatchNormalization()(x)
        x = sep_conv1d(f, ksize, padding="valid")(x)
        x = layers.BatchNormalization()(x)
        ksize = min(x.shape.as_list()[1:] + [ksize])
        # if ksize > 1:
        #     x = layers.AveragePooling1D(pool_size=4)(x)
        ksize = min(x.shape.as_list()[1:] + [ksize])
        # x = layers.Dropout(1.0 / 16.0)(x)
        # f *= 2
        i += 1

    # x = layers.BatchNormalization()(x)
    # x = layers.Reshape((x.shape[-1], 1))(x)
    # x = layers.LocallyConnected1D(64, kernel_size=x.shape[-2], activation="relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(1.0 / 2.0)(x)
    x = layers.Dense(
        dense_size,
        activation="relu",
        bias_initializer=rnd_uniform,
        bias_regularizer=l2_reg,
        kernel_initializer=rnd_uniform,
        kernel_regularizer=l2_reg,
    )(x)
    outputs = layers.Dense(
        output_shape[0],
        activation="softmax",
        bias_initializer=rnd_uniform,
        bias_regularizer=l2_reg,
        kernel_initializer=rnd_uniform,
        kernel_regularizer=l2_reg,
        name="outputs",
    )(x)

    model = keras.Model(inputs, outputs)

    ROC = keras.metrics.AUC()
    model.compile(
        # loss=keras.losses.CategoricalCrossentropy(),
        # loss=keras.losses.MeanSquaredError(),
        loss=keras.losses.CosineSimilarity(),
        # loss=keras.losses.KLDivergence(),
        # loss=keras.losses.MeanAbsoluteError(),
        # loss=abs_cat_loss,
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        metrics=[ROC],
    )
    print(model.summary())
    return model


def conv1D(input_shape, output_shape, filters, kernel_size, dense_size):
    max_filters = 2 ** 10
    l1_reg = keras.regularizers.l1(l=1e-5)
    l2_reg = keras.regularizers.l2(l=1e-5)
    w = pywt.Wavelet("db8")
    filter_len = w.dec_len
    mode = "zero"
    level = 4
    wavelet_len = pywt.dwt_coeff_len(input_shape[0], filter_len, mode=mode)
    inputs = keras.Input(shape=(wavelet_len, 1), name="inputs")
    x = inputs
    x = layers.LayerNormalization(axis=[1, 2])(x)
    # x = layers.Dropout(1.0 / 16.0)(x)
    ksize = kernel_size
    f = filters
    i = 0
    while ksize > 1 and i < 64:
        x = layers.SeparableConv1D(
            min(max_filters, f),
            ksize,
            input_shape=input_shape,
            padding="valid",
            # strides=ksize,
            # activation="relu",
            bias_initializer=keras.initializers.RandomNormal(),
            bias_regularizer=l1_reg,
            kernel_initializer=keras.initializers.RandomNormal(),
            kernel_regularizer=l1_reg,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation="softsign")(x)
        # if ksize > 1:
        #     x = layers.MaxPool1D(pool_size=2)(x)
        ksize = min(x.shape.as_list()[1:] + [ksize])
        f *= 2
        i += 1

    # x = layers.Reshape((x.shape[1] * x.shape[3], 1))(x)
    # x = layers.Conv1D(filters, kernel_size, padding="valid", activation="relu")(x)
    # x = layers.MaxPool1D(pool_size=kernel_size)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(1.0 / 4.0)(x)
    x = layers.Dense(
        dense_size,
        activation="relu",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg,
    )(x)
    outputs = layers.Dense(
        output_shape[0],
        activation="softmax",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg,
        name="outputs",
    )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        # loss=keras.losses.CategoricalCrossentropy(),
        # loss=keras.losses.MeanSquaredError(),
        loss=keras.losses.CosineSimilarity(),
        # loss=keras.losses.KLDivergence(),
        # loss=keras.losses.MeanAbsoluteError(),
        # loss=abs_cat_loss,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    print(model.summary())
    return model


def conv3D(input_shape, output_shape, filters, kernel_size, dense_size):
    # [32,32,64,64,128,128,256,256,256,256,256,256,256,512]:  # 13
    # [16,16,32,32,64,64,128,128,256,256,512,512,1024,1024,1024]:  # 10
    max_filters = 2 ** 10
    l1_reg = keras.regularizers.l1(l=1e-8)
    l2_reg = keras.regularizers.l2(l=1e-8)
    inputs = keras.Input(shape=input_shape, name="inputs")
    x = inputs

    # x = layers.experimental.preprocessing.Normalization()(x)
    # x = layers.LocallyConnected1D(1, kernel_size=1)(x)
    # x = layers.Reshape((8, 8, 8, 1))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.LayerNormalization(axis=[1, 2, 3, 4])(x)

    ksize = kernel_size
    f = filters
    i = 0
    while ksize > 1 and i < 64:
        x = layers.Dropout(1.0 / 16.0)(x)
        x = layers.Conv3D(
            min(max_filters, f),
            ksize,
            input_shape=input_shape,
            padding="valid",
            # strides=ksize,
            activation="softsign",
            bias_initializer=keras.initializers.RandomNormal(),
            bias_regularizer=l1_reg,
            kernel_initializer=keras.initializers.RandomNormal(),
            kernel_regularizer=l1_reg,
        )(x)
        x = layers.BatchNormalization()(x)
        ksize = min(x.shape.as_list()[1:] + [ksize])
        f *= 2
        i += 1

    # x = layers.LayerNormalization()(x)

    # outputs = layers.LocallyConnected1D(
    #     output_shape[0], kernel_size=x.shape[-2], activation="softmax"
    # )(x)

    x = layers.Flatten()(x)

    outputs = layers.Dense(
        output_shape[0],
        activation="softmax",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg,
        name="outputs",
    )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        # loss=keras.losses.CosineSimilarity(),
        # loss=keras.losses.KLDivergence(),
        # loss=keras.losses.MeanAbsoluteError(),
        # loss=abs_cat_loss,
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=["mean_absolute_error"],
    )
    print(model.summary())
    return model


def multiConv2D(input_shape, output_shape, filters, kernel_size, dense_size):
    l1_reg = keras.regularizers.l1(l=1e-5)
    l2_reg = keras.regularizers.l2(l=1e-5)
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # [32,32,64,64,128,128,256,256,256,256,256,256,256,512]:  # 13
    # [16,16,32,32,64,64,128,128,256,256,512,512,1024,1024,1024]:  # 10
    ksize = min([x.shape[1], x.shape[2], kernel_size])
    i = 0
    l = []
    while ksize > 1:
        x = layers.SeparableConv2D(
            filters,
            ksize,
            padding="valid",
            bias_initializer=keras.initializers.RandomNormal(),
            bias_regularizer=l2_reg,
            kernel_initializer=keras.initializers.RandomNormal(),
            kernel_regularizer=l2_reg,
        )(x)
        x = layers.Activation("softsign")(x)
        x = layers.BatchNormalization()(x)
        s = layers.MaxPool2D(pool_size=(x.shape[1], x.shape[2]), strides=(1, 1))(x)
        l.append(s)
        ksize = min([x.shape[1], x.shape[2], kernel_size])
        i += 1

    m = layers.Concatenate()(l)
    x = layers.Dropout(0.1)(m)
    x = layers.Flatten()(x)
    x = layers.Dense(
        dense_size * 8,
        activation="softsign",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg,
    )(x)
    x = layers.Dense(
        dense_size,
        activation="softsign",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg,
    )(x)
    outputs = layers.Dense(
        output_shape[0],
        activation="softmax",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg,
    )(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        loss=keras.losses.CosineSimilarity(),
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        metrics=["accuracy", "AUC", "mean_absolute_error"],
    )
    print(model.summary())
    return model


def conv2D(input_shape, output_shape, filters, kernel_size, dense_size):
    # [32,32,64,64,128,128,256,256,256,256,256,256,256,512]:  # 13
    # [16,16,32,32,64,64,128,128,256,256,512,512,1024,1024,1024]:  # 10
    max_filters = 2 ** 9
    l1_reg = keras.regularizers.l1(l=1e-6)
    l2_reg = keras.regularizers.l2(l=1e-6)
    inputs = keras.Input(shape=input_shape, name="inputs")
    ksize = kernel_size
    x = inputs
    f = filters
    i = 0
    while ksize > 1 and i < 16:
        i += 1
        x = layers.SeparableConv1D(
            min(max_filters, f),
            ksize,
            padding="valid",
            activation="relu",
            bias_initializer=keras.initializers.RandomNormal(),
            bias_regularizer=l2_reg,
            kernel_initializer=keras.initializers.RandomNormal(),
            kernel_regularizer=l2_reg,
        )(x)
        ksize = min(x.shape.as_list()[1:] + [ksize])
        x = layers.BatchNormalization()(x)
        # if x.shape[-2] >= 3:
        #     x = layers.AveragePooling1D(2)(x)
        x = layers.Dropout(1.0 / 64.0)(x)
        f += 16

    # x = layers.BatchNormalization()(x)
    # x = layers.LocallyConnected1D(8, kernel_size=1)(x)
    # for i in range(8):
    #     # x = layers.Reshape((x.shape[-1], 1))(x)
    #     x = layers.LSTM(filters, return_sequences=True)(x)

    x = layers.Dropout(1.0 / 4.0)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(
        output_shape[0] * 8,
        activation="softsign",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg,
    )(x)
    x = layers.Dense(
        output_shape[0] * 4,
        activation="softsign",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg,
        name="dense_2",
    )(x)
    outputs = layers.Dense(
        output_shape[0],
        activation="softmax",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l2_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l2_reg,
        name="outputs",
    )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        loss=keras.losses.CosineSimilarity(),
        # loss=keras.losses.KLDivergence(),
        # los s=keras.losses.MeanAbsoluteError(),
        # loss=abs_cat_loss,
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        metrics=["categorical_accuracy", "mean_absolute_error"],
    )
    print(model.summary())
    return model
