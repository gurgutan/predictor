import math
from matplotlib.pyplot import axes
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, metrics
from tensorflow.keras.layers import Dense, LSTM, Concatenate, Lambda
from tensorflow.keras.layers import Reshape, BatchNormalization, Conv1D
from tensorflow.keras.layers import Input, Flatten, LeakyReLU, SeparableConv1D
from tensorflow.keras.layers import Dropout, LayerNormalization, MultiHeadAttention

# from keras.layers import Sequential
# from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.ops.gen_math_ops import Mul

# from tensorflow.keras.utils import to_categorical

from rbflayer import RBFLayer


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


def f_std(z):
    return tf.math.reduce_std(z, 1, keepdims=True)


def f_mean(z):
    return tf.math.reduce_mean(z, 1, keepdims=True)


def f_vrange(z):
    return tf.math.reduce_max(z, 1, True) - tf.math.reduce_min(z, 1, True)


def f_log(x):
    return tf.math.log(tf.exp(1.0) + tf.abs(x))


def f_logtanh(x):
    return tf.math.log(tf.exp(1.0) + tf.abs(x)) * tf.tanh(x)  # type: ignore


def f_dct(x):
    return tf.signal.dct(x, n=64, norm="ortho")


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
        value_min=-2.0,
        value_max=2.0,
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


class TanhE(losses.Loss):
    def __init__(
        self, multiplier=8.0, reduction=losses.Reduction.AUTO, name="tanhe",
    ) -> None:
        self.multiplier = multiplier
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        tanh_y_pred = tf.nn.tanh(y_pred * self.multiplier)
        tanh_y_true = tf.nn.tanh(y_true * self.multiplier)
        return losses.mean_squared_error(tanh_y_true, tanh_y_pred)


class ClippedMAE(losses.Loss):
    def __init__(
        self,
        value_min=-1.0,
        value_max=1.0,
        reduction=losses.Reduction.AUTO,
        name="clipped_mae",
    ) -> None:
        super().__init__(reduction=reduction, name=name)
        self.value_min = value_min
        self.value_max = value_max

    def call(self, y_true, y_pred):
        clipped_y_pred = tf.clip_by_value(y_pred, self.value_min, self.value_max)
        clipped_y_true = tf.clip_by_value(y_true, self.value_min, self.value_max)
        return losses.mean_absolute_error(clipped_y_true, clipped_y_pred)


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


class ClippedCSE(tf.keras.losses.Loss):
    def __init__(
        self,
        value_min=-1.0,
        value_max=1.0,
        count=8,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="clipped_cse",
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
        indices = tf.cast((clipped_y_true - self.value_min) * self.step, tf.int32)
        indices = tf.reshape(indices, [-1])
        y_true_batch = tf.zeros_like(y_pred)
        # np.arange(y_pred.shape[0])
        y_true_batch[1, indices] = 1.0
        return tf.losses.cosine_similarity(y_true_batch, clipped_y_pred)


def ConvAdaptiveKernelSize(
    x, activation, filters=8, kernel_size=2, dropout=0.5, name=""
):
    k_size = kernel_size if x.shape[-2] >= kernel_size else x.shape[-2]
    l2 = keras.regularizers.l2(1e-10)  # type: ignore
    x = Conv1D(filters, k_size, padding="valid")(x)
    x = LayerNormalization()(x)
    # x = BatchNormalization()(x)
    x = Lambda(activation)(x)
    # x = Dropout(rate=dropout)(x)
    return x


def transformer_encoder(inputs, head_size, num_heads, ff_dim, d=0.0):
    l2 = keras.regularizers.L2(l2=1e-8)  # type: ignore
    x = LayerNormalization(epsilon=1e-8)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=d)(
        x, inputs
    )
    # x = Dropout(d)(x)
    x = LayerNormalization(epsilon=1e-8)(x)
    res = x + inputs
    x = Conv1D(
        filters=ff_dim,
        kernel_size=1,
        activation="relu",
        kernel_regularizer=l2,
        bias_regularizer=l2,
    )(res)
    x = Dropout(d)(x)
    x = Conv1D(
        filters=inputs.shape[-1],
        kernel_size=1,
        kernel_regularizer=l2,
        bias_regularizer=l2,
    )(x)
    x = LayerNormalization()(x)
    return x + res


def t1(
    input_width,
    out_width,
    columns=16,
    lr=1e-2,
    min_v=-2.0,
    max_v=2.0,
    training=True,
    dropout=0.5,
    name="t1",
):
    filters = 1024
    head_size = 64
    num_heads = 64
    rows = 6
    name = f"t-{filters}-{head_size}-{num_heads}-{rows}"
    inputs = Input(shape=(input_width,))
    x = Lambda(f_dct, name=f"dct")(inputs, input_width)
    x = Dense(64, name=f"d-0")(x)
    x = Reshape((1, -1))(x)
    for i in range(rows):
        x = transformer_encoder(x, head_size, num_heads, filters, dropout)
    x = Flatten()(x)
    x = Dense(num_heads, name=f"d-1")(x)
    rows = 4
    units = 16
    z = [Dense(units, name=f"d-in{c}-{0}")(x) for c in range(columns)]
    z = [Lambda(f_logtanh, name=f"logtanh-in-{c}")(z[c]) for c in range(columns)]
    for c in range(columns):
        for r in range(rows - 1):
            z[c] = Dense(units, name=f"d{c}-{r}")(z[c])
            z[c] = BatchNormalization()(z[c])
            z[c] = Lambda(f_logtanh, name=f"logtanh-{c}-{r}")(z[c])
        z[c] = Dense(out_width)(z[c])
        # z[c] = Lambda(f_logtanh)(z[c])
    x = Concatenate()(z)
    x = Dense(out_width)(x)
    # x = Lambda(f_logtanh)(x)
    outputs = x
    model = keras.Model(inputs, outputs, name=name)
    MAE = keras.metrics.MeanAbsoluteError()
    CMSE = ClippedMSE(min_v, max_v)
    # CMAE = ClippedMAE(min_v, max_v)
    keras.mixed_precision.set_global_policy("mixed_float16")  # type: ignore
    model.compile(
        # loss=keras.losses.LogCosh(),
        # loss=keras.losses.MeanSquaredLogarithmicError(),
        # loss=keras.losses.MeanSquaredError(),
        loss=CMSE,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[MAE],
    )
    return model


def red(
    input_width,
    out_width,
    columns=16,
    lr=1e-2,
    min_v=-2.0,
    max_v=2.0,
    training=True,
    name="red",
):
    if training:
        dropout = 1.0 / 256.0
    else:
        dropout = 0
    init = keras.initializers.RandomUniform(-1024, 1024)
    l2 = keras.regularizers.L2(l2=1e-10)  # type: ignore
    dct_length = input_width

    # def f_dct(x): return tf.signal.mdct(
    # x, frame_length = 8, norm = 'ortho', pad_end = True)
    n = int(math.log2(input_width))
    filters = 32
    inputs = Input(shape=(input_width,))
    # key = [Lambda(lambda z: z[:, -(2 ** (i+1)) :])(inputs) for i in range(n)]
    # m = [Lambda(f_mean, name=f"mean{i}")(key[i]) for i in range(n)]
    # m = Concatenate(name=f"concat_means")(m)
    # m = Dense(filters)(m)
    # m = Reshape((1, -1))(m)
    # f = Dense(64)(inputs)
    f = Lambda(f_dct, name=f"dct")(inputs, dct_length)
    f = Reshape((-1, 1))(f)
    i = 1
    while f.shape[-2] > 1:
        i = i + 1
        f = ConvAdaptiveKernelSize(f, tf.nn.tanh, filters, 16, dropout, name=f"{i}")
    # f = Dropout(rate=dropout)(f)
    # x = Multiply()([m, f])
    # x = Flatten()(f)
    # x = Dense(64)(x)
    x = Reshape((-1, 1))(f)
    x = LSTM(128, return_sequences=True, dropout=dropout, name="lstm-1")(x)
    x = Flatten()(x)
    x = Dense(32, name=f"d-in-0")(x)
    rows_count = 4
    units = 32
    z = [Dense(units, name=f"d-in{c}-{0}")(x) for c in range(columns)]
    z = [Lambda(f_logtanh, name=f"logtanh-in-{c}")(z[c]) for c in range(columns)]
    for c in range(columns):
        for r in range(rows_count - 1):
            z[c] = Dense(units, name=f"d{c}-{r}")(z[c])
            z[c] = BatchNormalization()(z[c])
            z[c] = Lambda(f_logtanh, name=f"logtanh-{c}-{r}")(z[c])
        z[c] = Dense(out_width)(z[c])
        # z[c] = Lambda(f_logtanh)(z[c])
    x = Concatenate()(z)
    x = Dense(out_width)(x)
    # x = Lambda(f_logtanh)(x)
    outputs = x
    model = keras.Model(inputs, outputs, name=name)
    MAE = keras.metrics.MeanAbsoluteError()
    CMSE = ClippedMSE(min_v, max_v)
    CMAE = ClippedMAE(min_v, max_v)
    model.compile(
        # loss=keras.losses.Huber(),
        # loss=keras.losses.MeanSquaredError(),
        loss=CMSE,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[MAE],
    )
    return model


def red1(
    input_width,
    out_width,
    columns=16,
    lr=1e-2,
    min_v=-2,
    max_v=2,
    training=True,
    name="red",
):
    if training:
        dropout = 1.0 / 64.0
    else:
        dropout = 0
    # диапазон инициализации
    init = keras.initializers.RandomUniform(-256, 256)
    # регуляризатор
    l2 = keras.regularizers.L2(l2=1e-8)
    # длина сэмпла - результата преобразованного dct
    dct_length = 32  # input_width
    n = int(math.log2(input_width))
    filters = 32
    inputs = Input(shape=(input_width,))
    f = LayerNormalization()(inputs)
    f = Lambda(f_dct, name=f"dct")(f)
    f = Reshape((1, -1))(f)
    for i in range(8):
        f = transformer_encoder(f, 64, 64, filters, d=dropout)
    # x = Reshape((-1, 1))(f)
    # x = LSTM(64, return_sequences=True, dropout=dropout)(x)
    x = Flatten()(f)
    x = Dense(32)(x)
    rows_count = 3
    units = 16
    z = [Dense(units, name=f"d-in{c}-{0}")(x) for c in range(columns)]
    z = [Lambda(f_logtanh)(z[c]) for c in range(columns)]
    for c in range(columns):
        for r in range(rows_count - 1):
            z[c] = Dense(
                units, kernel_regularizer=l2, bias_regularizer=l2, name=f"d{c}-{r}"
            )(z[c])
            z[c] = LayerNormalization()(z[c])
            z[c] = Activation("relu")(z[c])
            z[c] = Dropout(rate=1.0 / 256.0)(z[c], training=False)
        z[c] = Dense(out_width)(z[c])
        # z[c] = Lambda(f_logtanh)(z[c])
    x = Concatenate()(z)
    x = Dense(out_width)(x)
    # x = Lambda(f_logtanh)(x)
    outputs = x
    model = keras.Model(inputs, outputs, name=name)
    MAE = keras.metrics.MeanAbsoluteError()
    CMSE = ClippedMSE(min_v, max_v)
    CMAE = ClippedMAE(min_v, max_v)
    model.compile(
        # loss=keras.losses.Huber(),
        # loss=keras.losses.MeanSquaredError(),
        loss=CMSE,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[MAE],
    )
    return model


def mh_att(
    input_width,
    out_width,
    columns=16,
    lr=1e-4,
    min_v=-2.0,
    max_v=2.0,
    training=True,
    name="mhatt",
):
    if training:
        dropout = 1.0 / 64.0
    else:
        dropout = 0.0

    init_scale = 2 ** 8
    kern_init = keras.initializers.RandomUniform(-init_scale, init_scale)
    l2 = keras.regularizers.L2(l2=1e-10)  # type: ignore
    n = int(math.log2(input_width))
    sample_width = min(4, input_width)
    inputs = Input(shape=(input_width,))
    y = BatchNormalization()(inputs)
    x = [Lambda(lambda z: z[:, -(2 ** (i + 1)) :])(y) for i in range(n)]
    u = Lambda(lambda z: z[:, -sample_width:])(y)
    u = Dense(n)(u)
    u = Reshape((-1, 1))(u)
    means = [Lambda(f_mean, name=f"mean{i}")(x[i]) for i in range(n)]
    m = Concatenate(1, name=f"concat_means")(means)
    m = Reshape((-1, 1))(m)
    stds = [Lambda(f_std, name=f"std{i}")(x[i]) for i in range(n)]
    s = Concatenate(1, name=f"concat_stds")(stds)
    s = Reshape((-1, 1))(s)
    filters = 16
    head_size = 32
    num_heads = 256
    kernel_size = 2
    for i in range(6):
        m = ConvAdaptiveKernelSize(m, tf.nn.tanh, filters, kernel_size, kern_init)
        s = ConvAdaptiveKernelSize(s, tf.nn.relu, filters, kernel_size, kern_init)
        u = ConvAdaptiveKernelSize(u, tf.nn.tanh, filters, kernel_size, kern_init)
    x = Concatenate(axis=-2)([s, m])
    # x = BatchNormalization()(x)
    for i in range(4):
        x = transformer_encoder(x, u, head_size, num_heads, filters, dropout)
    x = Flatten()(x)
    rows_count = 4
    units = 16
    z = [Dense(units, name=f"d-in{c}-{0}")(x) for c in range(columns)]
    z = [Lambda(f_logtanh)(z[c]) for c in range(columns)]
    for c in range(columns):
        for r in range(rows_count - 1):
            z[c] = Dense(units, name=f"d{c}-{r}")(z[c])
            z[c] = BatchNormalization()(z[c])
            z[c] = Lambda(f_logtanh)(z[c])
        z[c] = Dense(out_width)(z[c])
        # z[c] = Lambda(f_logtanh)(z[c])
    x = Concatenate()(z)
    x = Dense(out_width)(x)
    # x = Lambda(f_logtanh)(x)
    outputs = x
    model = keras.Model(inputs, outputs, name=name)
    MAE = keras.metrics.MeanAbsoluteError()
    CMSE = ClippedMSE(min_v, max_v)
    CMAE = ClippedMAE(min_v, max_v)
    model.compile(
        # loss=keras.losses.Huber(),
        # loss=keras.losses.MeanSquaredError(),
        loss=CMSE,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[MAE],
    )
    return model


def dense_boost(
    input_width, out_width, columns=4, lr=1e-3, min_v=-1, max_v=1, name="d-boost"
):
    init_scale = 2 ** 10
    kernel_init = keras.initializers.RandomUniform(-init_scale, init_scale)
    # l2 = keras.regularizers.l2(1e-10)
    def f_std(z):
        return tf.math.reduce_std(z, 1, keepdims=True)

    def f_mean(z):
        return tf.math.reduce_mean(z, 1, keepdims=True)

    def f_vrange(z):
        return tf.math.reduce_max(z, 1, keepdims=True) - tf.math.reduce_min(
            z, 1, keepdims=True
        )

    def f_logtanh(x):
        return tf.math.log(tf.exp(1.0) + tf.abs(x)) * tf.tanh(x)

    def rad(x):
        return 2 - tf.sqrt(1 + tf.math.square(x))

    # rad2 = lambda x: 1 - tf.math.log1p(tf.sqrt(tf.abs(x)))
    # rad3 = lambda x: 2 - tf.math.sqrt(1 + x ** 2)
    # rsig = x / (1.0 + 4 * tf.sqrt(tf.abs(x)))
    n = int(math.log2(input_width))
    slope = 1.0 / (2 ** 10)
    k_size = 3
    sample_width = min(4, input_width)
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
    x = Concatenate(axis=-2)([m, s, t, u])
    x = BatchNormalization()(x)
    x = Reshape((1, -1))(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Flatten()(x)
    rows = 4
    units = 16
    z = [Dense(units, name=f"d-{c}-in")(x) for c in range(columns)]
    z = [Lambda(f_logtanh)(z[c]) for c in range(columns)]
    for c in range(columns):
        for r in range(rows):
            z[c] = Dense(units, name=f"d{c}-{r}")(z[c])
            z[c] = Lambda(f_logtanh)(z[c])
        # z[c] = Softmax()(z[c])
        z[c] = Dense(out_width, name=f"d-{c}-out")(z[c])
        z[c] = Lambda(f_logtanh)(z[c])
    x = Concatenate()(z)
    x = Dense(out_width)(x)
    outputs = x
    model = keras.Model(inputs, outputs, name=name)
    MAE = keras.metrics.MeanAbsoluteError()
    CMSE = ClippedMSE(min_v, max_v)
    model.compile(
        # loss=keras.losses.Huber(),
        # loss=keras.losses.MeanSquaredError(),
        loss=CMSE,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[MAE],
    )
    return model


def dense_att(
    input_width,
    out_width,
    columns=4,
    lr=1e-4,
    min_v=-2.0,
    max_v=2.0,
    training=True,
    name="dense-att",
):
    init_scale = 2 ** 12
    init = keras.initializers.RandomUniform(-init_scale, init_scale)
    # l2 = keras.regularizers.l2(1e-10)
    # rad = lambda x: 2 - tf.sqrt(1 + tf.math.square(x))
    # rad2 = lambda x: 1 - tf.math.log1p(tf.sqrt(tf.abs(x)))
    # rad3 = lambda x: 2 - tf.math.sqrt(1 + x ** 2)
    # rsig = x / (1.0 + 4 * tf.sqrt(tf.abs(x)))
    n = int(math.log2(input_width))
    slope = 1.0 / (2 ** 10)
    kernel_size = 3
    dropout = 0.0
    if training:
        dropout = 0.2
    sample_width = min(4, input_width)
    inputs = Input(shape=(input_width,))
    x = inputs
    u = Lambda(lambda z: z[:, -sample_width:])(x)
    u = Dense(n)(u)
    u = Reshape((-1, 1))(u)
    x = [Lambda(lambda z: z[:, -(2 ** (i + 1)) :])(inputs) for i in range(n)]
    m = [Lambda(f_mean, name=f"mean{i}")(2 ** i * x[i]) for i in range(n)]
    m = Concatenate(1, name=f"concat_means")(m)
    m = Reshape((-1, 1))(m)
    s = [Lambda(f_std, name=f"std{i}")(2 ** i * x[i]) for i in range(n)]
    s = Concatenate(1, name=f"concat_stds")(s)
    s = Reshape((-1, 1))(s)

    filters = 256
    for i in range(4):
        m = ConvAdaptiveKernelSize(m, tf.nn.tanh, filters, kernel_size, init)
        s = ConvAdaptiveKernelSize(s, tf.nn.tanh, filters, kernel_size, init)
        u = ConvAdaptiveKernelSize(u, tf.nn.tanh, filters, kernel_size, init)
    x = Concatenate(axis=-2)([m, s])
    x = BatchNormalization()(x)
    # x = Reshape((-1, 1))(x)
    x = MultiHeadAttention(128, 64, dropout=dropout)(x, u)
    x = Flatten()(x)
    rows = 4
    units = 32
    z = [Dense(units, name=f"d-{c}-in")(x) for c in range(columns)]
    z = [Lambda(f_logtanh)(z[c]) for c in range(columns)]
    for c in range(columns):
        for r in range(rows):
            z[c] = Dense(units, name=f"d{c}-{r}")(z[c])
            z[c] = Lambda(f_logtanh)(z[c])
        # z[c] = Softmax()(z[c])
        z[c] = Dense(out_width, name=f"d-{c}-out")(z[c])
        # z[c] = Lambda(f_logtanh)(z[c])
    x = Concatenate()(z)
    x = Dense(out_width)(x)
    outputs = x
    model = keras.Model(inputs, outputs, name=name)
    MAE = keras.metrics.MeanAbsoluteError()
    CMSE = ClippedMSE(min_v, max_v)
    model.compile(
        # loss=keras.losses.Huber(),
        # loss=keras.losses.MeanSquaredError(),
        loss=CMSE,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[MAE],
    )
    return model


def MultiKernel(num_heads=1, head_size=8, out_size=8, dropout=0.1, activation="relu"):
    kernels = [Dense(head_size, activation=activation) for i in range(num_heads)]

    def concat(x):
        return Concatenate()([kernels[i](x) for i in range(num_heads)])

    def drop(x):
        return Dropout(dropout)(x)

    def dense(x):
        return Dense(out_size, activation=activation)(drop(concat(x)))

    return dense


def tired(
    input_width,
    out_width,
    columns=16,
    lr=1e-2,
    min_v=-2,
    max_v=2,
    training=True,
    name="tired",
):
    if training:
        dropout = 1.0 / 16.0
    else:
        dropout = 0

    init = keras.initializers.RandomUniform(-1024, 1024)
    l2 = keras.regularizers.L2(l2=1e-10)
    dct_length = input_width

    def f_mean(z):
        return tf.math.reduce_mean(z, 1, keepdims=True)

    def f_logtanh(x):
        return tf.math.log(tf.exp(1.0) + tf.abs(x)) * tf.tanh(x)

    def f_dct(x):
        return tf.signal.dct(x, n=dct_length, norm="ortho")

    n = int(math.log2(input_width))
    filters = 32
    inputs = Input(shape=(input_width, 2))
    # hours = Dense(dct_length)(inputs[:, 1])
    # hours = Reshape((-1, 1))(hours)
    # rates = Dense(dct_length)(inputs[:, 0])
    # rates = Lambda(f_dct, name=f"dct")(rates)
    # rates = Reshape((-1, 1))(rates)
    hours = LayerNormalization()(inputs[:, 1])
    rates = Concatenate()([inputs[:, 0], hours])
    rates = Dense(1)(inputs)
    rates = Reshape((-1, dct_length))(rates)
    rates = Lambda(f_dct, name=f"dct")(rates)
    x = Reshape((-1, 1))(rates)
    i = 1
    while rates.shape[-2] > 1:
        i = i + 1
        x = ConvAdaptiveKernelSize(x, tf.nn.tanh, filters, 16, dropout, name=f"r{i}")
        # rates = ConvAdaptiveKernelSize(
        #     rates, tf.nn.tanh, filters, 16, dropout, name=f"r{i}")
        # hours = ConvAdaptiveKernelSize(
        #     hours, tf.nn.tanh, filters, 16, dropout, name=f"h{i}")
    # x = Add()([rates, hours])
    x = Reshape((-1, 1))(x)
    x = LSTM(64, return_sequences=True, dropout=dropout, name="lstm-1")(x)
    x = Flatten()(x)
    x = Dense(32, name=f"d-in-0")(x)
    rows_count = 4
    units = 16
    z = [Dense(units, name=f"d-in{c}-{0}")(x) for c in range(columns)]
    z = [Lambda(f_logtanh, name=f"logtanh-in-{c}")(z[c]) for c in range(columns)]
    for c in range(columns):
        for r in range(rows_count - 1):
            z[c] = Dense(units, name=f"d{c}-{r}")(z[c])
            z[c] = BatchNormalization()(z[c])
            z[c] = Lambda(f_logtanh, name=f"logtanh-{c}-{r}")(z[c])
        z[c] = Dense(out_width)(z[c])
    x = Concatenate()(z)
    x = Dense(out_width)(x)
    outputs = x
    model = keras.Model(inputs, outputs, name=name)
    MAE = keras.metrics.MeanAbsoluteError()
    CMSE = ClippedMSE(min_v, max_v)
    CMAE = ClippedMAE(min_v, max_v)
    model.compile(
        loss=CMSE, optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=[MAE],
    )
    return model


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
