import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def abs_cat_loss(y_true, y_pred):
    d = tf.abs(y_true - y_pred)
    return tf.reduce_mean(d, axis=-1)


def conv2D(input_shape, output_shape, filters, kernel_size, dense_size):
    max_filters = 256
    l1_reg = keras.regularizers.l1(l=1e-2)
    l2_reg = keras.regularizers.l2(l=1e-4)
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # for f in [i**2 for i in range(4, 1)]
    # [32,32,64,64,128,128,256,256,256,256,256,256,256,512]:  # 13
    # [16,16,32,32,64,64,128,128,256,256,512,512,1024,1024,1024]:  # 10
    ksize = min([x.shape[1], x.shape[2], kernel_size])
    f = 16
    while ksize > 1:
        x = layers.SeparableConv2D(
            min(max_filters, f),
            ksize,
            padding="valid",
            bias_initializer=keras.initializers.RandomNormal(),
            bias_regularizer=l2_reg,
            kernel_initializer=keras.initializers.RandomNormal(),
            kernel_regularizer=l2_reg,
        )(x)
        x = layers.Activation("softsign")(x)
        x = layers.BatchNormalization()(x)
        ksize = min([x.shape[1], x.shape[2], kernel_size])
        f += 16

    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    # x = layers.Dense(
    #     dense_size,
    #     activation="softsign",
    #     bias_initializer=keras.initializers.RandomNormal(),
    #     bias_regularizer=l1_reg,
    #     kernel_initializer=keras.initializers.RandomNormal(),
    #     kernel_regularizer=l1_reg,
    # )(x)

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
        # loss=keras.losses.MeanSquaredError(),
        loss=keras.losses.CosineSimilarity(),
        # loss=keras.losses.MeanAbsoluteError(
        #     reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        # ),
        # loss=keras.losses.Hinge(),
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        metrics=["accuracy", "AUC"],
    )
    print(model.summary())
    return model
