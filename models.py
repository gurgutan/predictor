import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow import keras
from tensorflow.python.keras.layers import LSTMV2


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


def dense_model(input_shape, output_shape, units, sections, train=True):
    l2 = keras.regularizers.l2(l=1e-3)
    dense_size = 256
    n = sections + 1
    inputs = Input(shape=input_shape, name="inputs")
    x = inputs
    # x = LayerNormalization(axis=[1, 2])(x)
    x = Conv1D(units, kernel_size=4, activation="relu")(x)
    x = Conv1D(256, kernel_size=4, activation="relu")(x)
    x = Conv1D(256, kernel_size=4, activation="relu")(x)
    x = Conv1D(128, kernel_size=4, activation="relu")(x)
    x = Conv1D(128, kernel_size=4, activation="relu")(x)
    x = Conv1D(64, kernel_size=4, activation="relu")(x)
    x = Conv1D(64, kernel_size=4, activation="relu")(x)
    # for i in range(4):
    #     # x = Reshape((-1, 1))(x)
    #     x = Conv1D(units, kernel_size=4, activation="relu", padding="same")(x)

    x = Flatten()(x)
    # x = Dense(256, "softmax")(x)
    x = Dense(dense_size, "tanh")(x)
    x = Dense(128, "tanh")(x)
    x = Dense(64, "tanh")(x)
    outputs = Dense(output_shape[-1])(x)
    # outputs = Activation("softsign")(x)
    model = keras.Model(inputs, outputs, name="dense2")
    MAE = keras.metrics.MeanAbsoluteError()
    model.compile(
        # loss=esum2,
        loss=keras.losses.MeanSquaredError(),
        # loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.SGD(learning_rate=1e-1),
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


def spectral(input_shape, units, width, depth):
    inputs = Input(shape=input_shape)
    z = [Conv1D(units, 2 ** i, strides=2 ** i) for i in range(width)]
    for j in range(depth):
        z = [Dense(units, activation=tf.nn.softsign)(z[i]) for i in range(width)]
    z = [Conv1D(units, kernel_size=z[i].shape[1])(z[i]) for i in range(width)]
    x = Concatenate(axis=1)(z)
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation("linear")(x)
    outputs = Reshape([1, -1])(x)
    model = keras.Model(inputs, outputs, name="spectral")
    AUC = keras.metrics.AUC()
    MAE = keras.metrics.MeanAbsoluteError()
    BC = keras.metrics.BinaryCrossentropy()
    model.compile(
        loss=mse_dir, optimizer=keras.optimizers.Adam(0.1), metrics=[MAE],
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
