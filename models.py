import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow import keras


def mse_dir(y_true, y_pred):
    penalty = 4.0
    cs = losses.cosine_similarity
    mse = losses.mean_squared_error
    mae = losses.mean_absolute_error
    y_true_sign = tf.math.softsign(y_true) * penalty
    y_pred_sign = tf.math.softsign(y_pred) * penalty
    return mse(y_true, y_pred) + mse(y_true_sign, y_pred_sign)


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


def trend_encoder(input_shape, units, sections):
    l2_reg = keras.regularizers.l2(l=1e-6)
    inputs = Input(shape=input_shape)
    x = inputs
    x = Lambda(lambda x: x[:, -(2 ** sections) :, :])(x)
    x = [AveragePooling1D(pool_size=2 ** i)(x) for i in range(sections)]
    x = [Lambda(lambda z: z * (2 ** i))(x[i]) for i in range(sections)]
    x = [Flatten()(x[i]) for i in range(sections)]
    x = [
        Dense(
            units=256,
            activation="relu",
            bias_regularizer=l2_reg,
            kernel_regularizer=l2_reg,
        )(x[i])
        for i in range(sections)
    ]
    x = [
        Dense(
            units=64,
            activation="relu",
            bias_regularizer=l2_reg,
            kernel_regularizer=l2_reg,
        )(x[i])
        for i in range(sections)
    ]
    x = Concatenate()(x)
    # x = Flatten()(x)
    x = Dense(
        units, activation="softmax", bias_regularizer=l2_reg, kernel_regularizer=l2_reg,
    )(x)
    x = Dense(
        256, activation="softsign", bias_regularizer=l2_reg, kernel_regularizer=l2_reg,
    )(x)
    x = Dense(
        64, activation="softsign", bias_regularizer=l2_reg, kernel_regularizer=l2_reg,
    )(x)
    x = Dropout(1 / 4)(x)
    x = Dense(input_shape[-1])(x)
    x = Activation(activation="linear")(x)
    model = keras.Model(inputs, x, name="trendencoder")
    MAE = keras.metrics.MeanAbsoluteError()
    RMSE = keras.metrics.RootMeanSquaredError()
    CS = keras.metrics.CosineSimilarity()
    model.compile(
        loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=[RMSE],
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
        loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        metrics=[RMSE],
    )
    print(model.summary())
    return model
