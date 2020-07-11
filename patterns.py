import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def abs_cat_loss(y_true, y_pred):
    d = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred))
    return d  # tf.reduce_mean(d, axis=-1)


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
    max_filters = 512
    l1_reg = keras.regularizers.l1(l=1e-6)
    l2_reg = keras.regularizers.l2(l=1e-6)
    inputs = keras.Input(shape=input_shape, name="inputs")
    x = inputs
    # [32,32,64,64,128,128,256,256,256,256,256,256,256,512]:  # 13
    # [16,16,32,32,64,64,128,128,256,256,512,512,1024,1024,1024]:  # 10
    ksize = min([x.shape[1], x.shape[2], kernel_size])
    f = filters
    i = 0
    while ksize > 1:
        i += 1
        x = layers.SeparableConv2D(
            min(max_filters, f),
            ksize,
            padding="valid",
            activation="relu",
            bias_initializer=keras.initializers.RandomNormal(),
            bias_regularizer=l2_reg,
            kernel_initializer=keras.initializers.RandomNormal(),
            name=f"conv2d_{str(i)}"
            # kernel_regularizer=l2_reg,
        )(x)
        x = layers.BatchNormalization(name=f"bnorma_{str(i)}")(x)
        ksize = min([x.shape[1], x.shape[2], ksize])
        f += 64

    x = layers.Reshape((x.shape[-1], 1), name="reshape")(x)
    x = layers.LocallyConnected1D(8, kernel_size=1, name="locconn1d")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(
        output_shape[0] * 8,
        activation="sigmoid",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg,
        name="dense_1",
    )(x)
    x = layers.Dense(
        output_shape[0] * 4,
        activation="sigmoid",
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
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg,
        name="outputs",
    )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        loss=keras.losses.CosineSimilarity(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy", "mean_absolute_error"],
    )
    print(model.summary())
    return model

