import math
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow import keras
from tensorflow.python.framework.tensor_util import SlowAppendBFloat16ArrayToTensorProto
from tensorflow.python.keras.layers import LSTMV2
from rbflayer import RBFLayer
import numpy as np


def mse_dir(y_true, y_pred):
    sign_penalty = 1.0
    mse = losses.mean_squared_error
    mae = losses.mean_absolute_error
    y_true_sign = sign_penalty * tf.math.softsign(y_true * sign_penalty)
    y_pred_sign = sign_penalty * tf.math.softsign(y_pred * sign_penalty)
    k = sign_penalty * 2 - y_true_sign * y_pred_sign
    return mse(y_true, y_pred) * k


def esum(y_true, y_pred):
    y_true_sign = tf.math.softsign(y_true)
    y_pred_sign = tf.math.softsign(y_pred)
    s = tf.math.reduce_sum(tf.math.square(y_true - y_pred), axis=-1)
    return tf.math.reduce_mean(s, 0)


def esum2(y_true, y_pred):
    mse = losses.mean_squared_error
    y_true_sign = tf.math.softsign(y_true)
    y_pred_sign = tf.math.softsign(y_pred)
    s = tf.math.reduce_sum(tf.math.abs(y_pred), axis=-1)
    m = tf.math.reduce_mean(s, 0)
    e = tf.exp(-m * m)
    return mse(y_true, y_pred) / tf.math.reduce_mean(s, 0)


def spectral_rbf(input_width, output_width):
    l2 = keras.regularizers.l2(1e-8)
    depth = 4
    units = 2 ** 8
    n = int(math.log2(input_width)) + 1
    k_size = 4
    sample_width = min(output_width, input_width)
    inputs = Input(shape=(input_width,))
    x = inputs
    u = Lambda(lambda z: z[:, -sample_width:])(x)
    u = Dense(n)(u)
    u = Reshape((-1, 1))(u)
    x = [Lambda(lambda z: 2 ** i * z[0][:, -(2 ** z[1]) :])((x, i)) for i in range(n)]
    means = [
        Lambda(lambda z: tf.math.reduce_mean(z, 1, keepdims=True))(x[i])
        for i in range(n)
    ]
    m = Concatenate(1, name=f"means")(means)
    m = Reshape((-1, 1))(m)
    stds = [
        Lambda(lambda z: tf.math.reduce_std(z, 1, keepdims=True))(x[i])
        for i in range(n)
    ]
    s = Concatenate(1, name=f"stds")(stds)
    s = Reshape((-1, 1))(s)
    r = [
        Lambda(
            lambda z: tf.math.reduce_max(z, 1, keepdims=True)
            - tf.math.reduce_min(z, 1, keepdims=True)
        )(x[i])
        for i in range(n)
    ]
    r = Concatenate(1, name=f"r")(r)
    r = Reshape((-1, 1))(r)

    x = Concatenate(axis=2)([m, s, r, u])
    for i in range(2):
        x = Conv1D(64, kernel_size=k_size, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    # x = Dropout(1 / 8)(x)
    # x = Reshape((1, -1))(x)
    # x = LSTM(128, return_sequences=True)(x)
    # x = LSTM(64, return_sequences=True)(x)
    # x = Flatten()(x)
    # x = [BatchNormalization()(x[i]) for i in range(output_width)]
    x = Flatten()(x)
    x = [Dense(units)(x) for i in range(output_width)]
    # x = [BatchNormalization()(x[i]) for i in range(output_width)]
    for i in range(depth):
        x = [Dense(units)(x[i]) for i in range(output_width)]
        x = [LeakyReLU(alpha=1 / 16)(x[i]) for i in range(output_width)]
    # x = [Dense(1024, "softmax")(x[i]) for i in range(output_width)]
    x = [Dense(1)(x[i]) for i in range(output_width)]
    x = Concatenate()(x)
    # x = kDropout(1 / 8)(x)
    # outputs = Dense(output_width, name="output")(x)
    outputs = x
    model = keras.Model(inputs, outputs, name="spectral2-2")
    MAE = keras.metrics.MeanAbsoluteError()
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        # loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(1e-5),
        metrics=[MAE],
    )
    return model


def rbf_dense(input_width, output_width, train=True):
    l2 = keras.regularizers.l2(l=1e-10)
    kernel_size = 4
    units = 256
    inputs = Input(shape=(input_width,), name="inputs")
    x = inputs
    # x = LayerNormalization(axis=[1, 2])(x)
    x = Flatten()(x)
    x = Dense(units)(x)
    x = RBFLayer(units)(x)
    x = Dense(units)(x)
    x = RBFLayer(units)(x)
    # x = Dense(units, "sigmoid")(x)
    # x = RBFLayerExp(256)(x)
    # x = RBFLayerExp(1024)(x)
    # x = Reshape((-1, 1))(x)
    # for i in range(6):
    #     x = Conv1D(
    #         min(2 ** (i + 3), 512),
    #         strides=2,
    #         kernel_size=kernel_size,
    #         activation="relu",
    #     )(x)
    # x = Conv1D(16, kernel_size=kernel_size, activation="relu")(x)
    # x = Conv1D(32, kernel_size=kernel_size, activation="relu")(x)
    # x = Conv1D(64, kernel_size=kernel_size, activation="relu")(x)
    # x = Conv1D(128, kernel_size=kernel_size, activation="relu")(x)
    # x = BatchNormalization(axis=[1, 2])(x)
    # x = Conv1D(256, kernel_size=kernel_size, activation="softsign",)(x)
    # x = BatchNormalization(axis=[1, 2])(x)
    # x = Flatten()(x)
    # x = Dropout(rate=1 / 4)(x)
    x = Dense(64, "tanh")(x)
    x = Dense(64, "tanh")(x)
    outputs = Dense(output_width)(x)
    model = keras.Model(inputs, outputs, name="rbf")
    MAE = keras.metrics.MeanAbsoluteError()
    model.compile(
        # loss=esum2,
        loss=keras.losses.MeanSquaredError(),
        # loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-7),
        metrics=[MAE],
    )
    print(model.summary())
    return model


def conv_model(input_shape, output_shape, units, train=True):
    l2 = keras.regularizers.l2(l=1e-14)
    kernel_size = 4
    inputs = Input(shape=input_shape, name="inputs")
    x = inputs
    x = BatchNormalization(axis=[1, 2])(x)
    x = SeparableConv1D(2 ** 10, kernel_size=kernel_size, activation="relu")(x)
    x = SeparableConv1D(2 ** 8, kernel_size=kernel_size, activation="relu")(x)
    x = SeparableConv1D(2 ** 8, kernel_size=kernel_size, activation="relu")(x)
    x = SeparableConv1D(2 ** 8, kernel_size=kernel_size, activation="tanh")(x)
    x = SeparableConv1D(2 ** 8, kernel_size=kernel_size, activation="tanh")(x)
    x = Flatten()(x)
    x = Dense(256, "softsign")(x)
    x = Dense(64, "softsign")(x)
    x = Dropout(rate=1 / 4)(x)
    outputs = Dense(output_shape[-1])(x)
    model = keras.Model(inputs, outputs, name="conv1")
    MAE = keras.metrics.MeanAbsoluteError()
    model.compile(
        # loss=esum2,
        loss=keras.losses.MeanSquaredError(),
        # loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-8),
        metrics=[MAE],
    )
    print(model.summary())
    return model


def dense_model(input_shape, output_shape, units, train=True):
    l2 = keras.regularizers.l2(l=1e-10)
    kernel_size = 4
    inputs = Input(shape=input_shape, name="inputs")
    x = inputs
    # x = LayerNormalization(axis=[1, 2])(x)
    # x = BatchNormalization(axis=[1, 2])(x)
    x = Conv1D(512, kernel_size=kernel_size, activation="relu",)(x)
    x = BatchNormalization(axis=[1, 2])(x)
    x = Conv1D(256, kernel_size=kernel_size, activation="relu",)(x)
    x = Flatten()(x)
    x = Dropout(rate=1 / 4)(x)
    x = Dense(256, "softsign")(x)
    x = Dense(64, "softsign")(x)
    outputs = Dense(output_shape[-1])(x)
    model = keras.Model(inputs, outputs, name="dense3")
    MAE = keras.metrics.MeanAbsoluteError()
    model.compile(
        # loss=esum2,
        loss=keras.losses.MeanSquaredError(),
        # loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
        metrics=[MAE],
    )
    print(model.summary())
    return model


def trend_encoder(input_shape, output_shape, units, sections, train=True):
    l2 = keras.regularizers.l2(l=1e-5)
    n = sections + 1
    inputs = Input(shape=input_shape, name="inputs")
    x = inputs
    x = Reshape(input_shape + (1,))(x)
    # std = Lambda(lambda z: tf.nn.moments(z, axes=[1, 2], keepdims=True)[1])(x)
    # x = LayerNormalization(axis=[1, 2])(x)
    x = Lambda(lambda z: z[:, -(2 ** sections) :, :])(x)
    x = [Lambda(lambda z: z[0][:, -(2 ** z[1]) :, :])((x, i)) for i in range(n)]
    # std = [
    #     Lambda(lambda z: tf.nn.moments(z, axes=[1, 2], keepdims=True)[1])(x[i])
    #     for i in range(n)
    # ]
    # x = [LSTM(16, return_sequences=True)(x[i]) for i in range(n)]
    # x = [Conv1D(8, 2 ** i, padding="same", activation="tanh")(x[i]) for i in range(n)]
    x = [
        Conv1D(2048, 2 ** i, padding="valid", activation="relu")(x[i]) for i in range(n)
    ]
    # y = [tf.math.divide(x[i], std[i]) for i in range(n)]
    x = Concatenate(axis=1)(x)
    x = Flatten()(x)
    x = Dense(units, "tanh")(x)
    x = Dense(256, "tanh")(x)
    # x = Dense(128, "tanh")(x)
    x = Dense(64, "tanh")(x)
    x = Dense(output_shape[-1])(x)
    outputs = x
    model = keras.Model(inputs, outputs, name="trendencoder3")
    MAE = keras.metrics.MeanAbsoluteError()
    model.compile(
        # loss=esum,
        loss=keras.losses.MeanSquaredError(),
        # loss=keras.losses.LogCosh(),
        # loss=keras.losses.KLDivergence(),
        # loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Nadam(learning_rate=1e-12),
        metrics=[MAE],
    )
    print(model.summary())
    return model


def lstm_block(input_shape, units, count=2):
    inputs = keras.Input(shape=input_shape, name="inputs")
    x = inputs
    for i in range(count - 1):
        x = LSTM(units, return_sequences=True)(x)
    x = LSTM(units, return_sequences=False)(x)
    x = Dense(64, activation="softsign",)(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    outputs = Activation("linear")(x)
    model = keras.Model(inputs, outputs)
    MAE = keras.metrics.MeanAbsoluteError()
    MSE = keras.metrics.MeanSquaredError()
    RMSE = keras.metrics.RootMeanSquaredError()
    model.compile(
        # loss=shifted_mse,
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        metrics=[RMSE],
    )
    print(model.summary())
    return model


def spectral(input_width, out_width, lr=1e-3):
    l2 = keras.regularizers.l2(1e-10)
    n = int(math.log2(input_width)) + 1
    slope = 1.0 / (2 ** 8)
    depth = 16
    units = 2 ** 6
    k_size = 3
    sample_width = min(8, input_width)
    inputs = Input(shape=(input_width,))
    x = inputs
    u = Lambda(lambda z: z[:, -sample_width:])(x)
    u = Dense(n)(u)
    u = Reshape((-1, 1))(u)
    x = [Lambda(lambda z: 2 ** i * z[0][:, -(2 ** z[1]) :])((x, i)) for i in range(n)]
    means = [
        Lambda(lambda z: tf.math.reduce_mean(z, 1, keepdims=True))(x[i])
        for i in range(n)
    ]
    m = Concatenate(1, name=f"means")(means)
    m = Reshape((-1, 1))(m)
    stds = [
        Lambda(lambda z: tf.math.reduce_std(z, 1, keepdims=True))(x[i])
        for i in range(n)
    ]
    s = Concatenate(1, name=f"stds")(stds)
    s = Reshape((-1, 1))(s)
    r = [
        Lambda(
            lambda z: tf.math.reduce_max(z, 1, keepdims=True)
            - tf.math.reduce_min(z, 1, keepdims=True)
        )(x[i])
        for i in range(n)
    ]
    r = Concatenate(1, name=f"r")(r)
    r = Reshape((-1, 1))(r)
    for i in range(4):
        m = Conv1D(64, kernel_size=k_size, padding="valid")(m)
        m = ReLU(negative_slope=slope)(m)
        s = Conv1D(64, kernel_size=k_size, padding="valid")(s)
        s = ReLU(negative_slope=slope)(s)
        r = Conv1D(64, kernel_size=k_size, padding="valid")(r)
        r = ReLU(negative_slope=slope)(r)
        u = Conv1D(64, kernel_size=k_size, padding="valid")(u)
        u = ReLU(negative_slope=slope)(u)

    x = Concatenate()([m, s, r, u])
    x = BatchNormalization()(x)
    x = Reshape((1, -1))(x)
    x = LSTM(256, return_sequences=True)(x)
    x = Flatten()(x)
    # x = Dropout(1 / 16)(x)
    x = [Dense(units)(x) for i in range(out_width)]
    for j in range(depth):
        x = [Dense(units, name=f"dense{j}-{i}")(x[i]) for i in range(out_width)]
        x = [
            ReLU(negative_slope=slope, name=f"relu{j}-{i}")(x[i])
            for i in range(out_width)
        ]
    x = [Dense(1, name=f"dense-out{i}")(x[i]) for i in range(out_width)]
    x = Concatenate()(x)
    outputs = x
    model = keras.Model(inputs, outputs, name="spectral4-2")
    MAE = keras.metrics.MeanAbsoluteError()
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        # loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[MAE],
    )
    return model


class ClippedMSE(losses.Loss):
    def __init__(
        self,
        value_min=-1.0,
        value_max=1.0,
        reduction=losses.Reduction.AUTO,
        name="clipped_mse",
    ) -> None:
        super().__init__(reduction=reduction, name=name)
        self.value_min = value_min
        self.value_max = value_max

    def call(self, y_true, y_pred):
        clipped_y_pred = tf.clip_by_value(y_pred, self.value_min, self.value_max)
        clipped_y_true = tf.clip_by_value(y_true, self.value_min, self.value_max)
        return losses.mean_squared_error(clipped_y_true, clipped_y_pred)


def dense_boost(
    input_width, out_width, columns=4, lr=1e-3, min_v=-1, max_v=1, name="dense-boost"
):
    init_scale = 2 ** 10
    kernel_init = keras.initializers.RandomUniform(-init_scale, init_scale)
    # l2 = keras.regularizers.l2(1e-10)
    f_std = lambda z: tf.math.reduce_std(z, 1, keepdims=True)
    f_mean = lambda z: tf.math.reduce_mean(z, 1, keepdims=True)
    f_vrange = lambda z: (
        tf.math.reduce_max(z, 1, keepdims=True)
        - tf.math.reduce_min(z, 1, keepdims=True)
    )
    f_logtanh = lambda x: tf.math.log(tf.exp(1.0) + tf.abs(x)) * tf.tanh(x)
    rad = lambda x: 2 - tf.sqrt(1 + tf.math.square(x))
    # rad2 = lambda x: 1 - tf.math.log1p(tf.sqrt(tf.abs(x)))
    # rad3 = lambda x: 2 - tf.math.sqrt(1 + x ** 2)
    # rsig = x / (1.0 + 4 * tf.sqrt(tf.abs(x)))
    n = int(math.log2(input_width))
    slope = 1.0 / (2 ** 10)
    rows = 8
    units = 64
    k_size = 3
    sample_width = min(2, input_width)
    inputs = Input(shape=(input_width,))
    x = inputs
    u = Lambda(lambda z: z[:, -sample_width:])(x)
    u = Dense(n)(u)
    # u = ReLU(negative_slope=slope)(u)
    u = Reshape((-1, 1))(u)
    x = [Lambda(lambda z: 2 ** i * z[0][:, -(2 ** z[1]) :])((x, i)) for i in range(n)]
    means = [Lambda(f_mean, name=f"mean{i}")(2 ** i * x[i]) for i in range(n)]
    m = Concatenate(1, name=f"concat_means")(means)
    m = Reshape((-1, 1))(m)
    stds = [Lambda(f_std, name=f"std{i}")(2 ** i * x[i]) for i in range(n)]
    s = Concatenate(1, name=f"concat_stds")(stds)
    s = Reshape((-1, 1))(s)
    t = [Lambda(f_vrange, name=f"range{i}",)(2 ** i * x[i]) for i in range(n)]
    t = Concatenate(1, name=f"concat_r")(t)
    t = Reshape((-1, 1))(t)
    filters = 64
    for i in range(3):
        m = Conv1D(filters, k_size, padding="valid", kernel_initializer=kernel_init)(m)
        m = Lambda(f_logtanh)(m)
        # m = ReLU(negative_slope=slope)(m)
        s = Conv1D(filters, k_size, padding="valid", kernel_initializer=kernel_init)(s)
        s = Lambda(f_logtanh)(s)
        # s = ReLU(negative_slope=slope)(s)
        t = Conv1D(filters, k_size, padding="valid", kernel_initializer=kernel_init)(t)
        t = Lambda(f_logtanh)(t)
        # t = ReLU(negative_slope=slope)(t)
        u = Conv1D(filters, k_size, padding="valid", kernel_initializer=kernel_init)(u)
        u = Lambda(f_logtanh)(u)
        # u = ReLU(negative_slope=slope)(u)
    x = Concatenate()([m, s, t, u])
    x = BatchNormalization()(x)
    x = Reshape((1, -1))(x)
    x = LSTM(256, return_sequences=True)(x)
    x = Flatten()(x)
    z = [x for k in range(columns)]
    out_range = range(out_width)
    for c in range(columns):
        z[c] = [Dense(units, name=f"d-in{c}-{i}")(z[c]) for i in out_range]
        for r in range(rows):
            z[c] = [Dense(units, name=f"d{c}-{r}-{i}")(z[c][i]) for i in out_range]
            z[c] = [Lambda(f_logtanh)(z[c][i]) for i in range(out_width)]
        z[c] = [Dense(1, name=f"d-out{c}-{i}")(z[c][i]) for i in out_range]
        # z[col] = [GaussianDropout(1 / 256)(z[col][i]) for i in out_range]
    for c in range(columns):
        z[c] = z[c][0] if len(z[c]) == 1 else Concatenate()(z[c])
    x = z[0] if len(z) == 1 else Concatenate()(z)
    x = Dense(out_width)(x)
    outputs = x
    model = keras.Model(inputs, outputs, name=name)
    MAE = keras.metrics.MeanAbsoluteError()
    CMSE = ClippedMSE(min_v, max_v)
    model.compile(
        # loss=keras.losses .Huber(),
        # loss=keras.losses.MeanSquaredError(),
        loss=CMSE,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[MAE],
    )
    return model


class ClippedSCE(losses.Loss):
    def __init__(
        self,
        value_min=-1.0,
        value_max=1.0,
        count=8,
        reduction=losses.Reduction.AUTO,
        name="clipped_sce",
    ) -> None:
        super().__init__(reduction=reduction, name=name)
        self.value_min = value_min
        self.value_max = value_max
        self.count = count
        self.step = (self.count - 1) / (self.value_max - self.value_min)

    def call(self, y_true, y_pred):
        clipped_y_pred = tf.clip_by_value(y_pred, self.value_min, self.value_max)
        clipped_y_true = tf.clip_by_value(y_true, self.value_min, self.value_max)
        # n_pred = (clipped_y_pred - self.value_min) * self.step
        n_true = (clipped_y_true - self.value_min) * self.step
        return losses.sparse_categorical_crossentropy(n_true, clipped_y_pred)


# class ClippedCSE(tf.keras.losses.Loss):
#     def __init__(
#         self,
#         value_min=-1.0,
#         value_max=1.0,
#         count=8,
#         reduction=tf.keras.losses.Reduction.AUTO,
#         name="clipped_cse",
#     ) -> None:
#         super().__init__(reduction=reduction, name=name)
#         self.value_min = value_min
#         self.value_max = value_max
#         self.count = count
#         self.step = (self.count - 1) / (self.value_max - self.value_min)

#     def call(self, y_true, y_pred):
#         clipped_y_pred = tf.clip_by_value(y_pred, self.value_min, self.value_max)
#         clipped_y_true = tf.clip_by_value(y_true, self.value_min, self.value_max)
#         # n_pred = (clipped_y_pred - self.value_min) * self.step
#         indices = tf.cast((clipped_y_true - self.value_min) * self.step, tf.int32)
#         indices = tf.reshape(indices, [-1])
#         y_true_batch = tf.zeros_like(y_pred)
#         # np.arange(y_pred.shape[0])
#         y_true_batch[1, indices] = 1.0
#         return tf.losses.cosine_similarity(y_true_batch, clipped_y_pred)


def prob_block(inputs, out_width, name="p"):
    # x = Flatten()(inputs)
    x = Dense(64)(inputs)
    x = Reshape((-1, 1))(x)
    for i, f in enumerate([16] * 4 + [32] * 4 + [64] * 8):
        x = Conv1D(f, 8, name=f"p-c{i}")(x)
        x = ReLU(negative_slope=1 / 2 ** 10)(x)
    x = Flatten()(x)
    x = Dense(64, name=f"p-d{64}")(x)
    x = ReLU(negative_slope=1 / 2 ** 10)(x)
    x = Dense(out_width, "softmax", name=name)(x)
    return x


def scored_boost(
    input_width,
    out_width,
    prob_width=8,
    columns=4,
    min_v=-4,
    max_v=4,
    lr=1e-3,
    name="scored-boost",
):
    init_scale = 2 ** 10
    kernel_init = keras.initializers.RandomUniform(-init_scale, init_scale)
    # l2 = keras.regularizers.l2(1e-10)
    f_std = lambda z: tf.math.reduce_std(z, 1, keepdims=True)
    f_mean = lambda z: tf.math.reduce_mean(z, 1, keepdims=True)
    f_vrange = lambda z: (
        tf.math.reduce_max(z, 1, keepdims=True)
        - tf.math.reduce_min(z, 1, keepdims=True)
    )
    f_logtanh = lambda x: tf.math.log(tf.exp(1.0) + tf.abs(x)) * tf.tanh(x)
    # rsig = x / (1.0 + 4 * tf.sqrt(tf.abs(x)))
    n = int(math.log2(input_width))
    rows = 8

    k_size = 3
    sample_width = min(2, input_width)
    inputs = Input(shape=(input_width,))
    x = inputs
    u = Lambda(lambda z: z[:, -sample_width:])(x)
    u = Dense(n)(u)
    u = Reshape((-1, 1))(u)
    x = [Lambda(lambda z: 2 ** i * z[0][:, -(2 ** z[1]) :])((x, i)) for i in range(n)]
    means = [Lambda(f_mean, name=f"mean{i}")(2 ** i * x[i]) for i in range(n)]
    m = Concatenate(1, name=f"concat_means")(means)
    m = Reshape((-1, 1))(m)
    stds = [Lambda(f_std, name=f"std{i}")(2 ** i * x[i]) for i in range(n)]
    s = Concatenate(1, name=f"concat_stds")(stds)
    s = Reshape((-1, 1))(s)
    t = [Lambda(f_vrange, name=f"range{i}",)(2 ** i * x[i]) for i in range(n)]
    t = Concatenate(1, name=f"concat_r")(t)
    t = Reshape((-1, 1))(t)
    for filters in [32, 64, 128]:
        m = Conv1D(filters, k_size, padding="valid", kernel_initializer=kernel_init)(m)
        m = Lambda(f_logtanh)(m)
        s = Conv1D(filters, k_size, padding="valid", kernel_initializer=kernel_init)(s)
        s = Lambda(f_logtanh)(s)
        t = Conv1D(filters, k_size, padding="valid", kernel_initializer=kernel_init)(t)
        t = Lambda(f_logtanh)(t)
        u = Conv1D(filters, k_size, padding="valid", kernel_initializer=kernel_init)(u)
        u = Lambda(f_logtanh)(u)
    x = Concatenate()([m, s, t, u])
    x = BatchNormalization()(x)

    # вероятностная
    p = prob_block(x, prob_width)

    # регрессионная
    units = 32
    x = Reshape((1, -1))(x)
    x = LSTM(256, return_sequences=True)(x)
    x = Flatten()(x)
    z = [x for k in range(columns)]
    out_range = range(out_width)
    for c in range(columns):
        z[c] = [Dense(units, name=f"d-in{c}-{i}")(z[c]) for i in out_range]
        for r in range(rows):
            z[c] = [Dense(units, name=f"d{c}-{r}-{i}")(z[c][i]) for i in out_range]
            z[c] = [Lambda(f_logtanh)(z[c][i]) for i in range(out_width)]
        z[c] = [Dense(1, name=f"d-out{c}-{i}")(z[c][i]) for i in out_range]
    for c in range(columns):
        z[c] = z[c][0] if len(z[c]) == 1 else Concatenate()(z[c])
    x = z[0] if len(z) == 1 else Concatenate()(z)
    x = Dense(out_width)(x)
    outputs = x
    model = keras.Model(inputs, outputs, name=name)
    x = Dense(out_width, name=f"v")(x)
    model = keras.Model(inputs, outputs=[x, p], name=name)
    MAE = keras.metrics.MeanAbsoluteError(name="mae")
    CMSE = ClippedMSE(min_v, max_v)
    SCCE = ClippedSCE(min_v, max_v, prob_width)
    # CCSE = ClippedCSE(min_v, max_v, prob_width)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={"v": CMSE, "p": SCCE},
        metrics={"v": MAE, "p": None},
        loss_weights={"v": 1.0, "p": 0.1},
    )
    return model


def multi_dense(input_shape, units, count=4):
    l2_reg = keras.regularizers.l2(l=1e-6)
    model = Sequential(
        [Input(shape=input_shape)]
        + [Dense(units=units, activation="relu") for i in range(count)]
        + [Flatten()]
        + [Dense(units=1)]
        + [Reshape([1, 1])],
        name="multi_dense",
    )
    AUC = keras.metrics.AUC()
    MAE = keras.metrics.MeanAbsoluteError()
    model.compile(
        loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
    )
    return model


def lstm_autoencoder(input_shape, units):
    model = Sequential(
        [
            LSTM(units=units, input_shape=input_shape),
            Dropout(rate=0.2),
            RepeatVector(n=input_shape[0]),
            LSTM(units=units, return_sequences=True),
            Dropout(rate=0.2),
            TimeDistributed(Dense(units=input_shape[1])),
        ]
    )
    model.compile(loss="mae", optimizer="Adam")
    return model
