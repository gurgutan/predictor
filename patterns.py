import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def conv2D_aka_inception(
    input_shape, output_shape, filters=32, kernel_size=4, dense_size=8
):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(filters, kernel_size, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256]:  # , 512, 728]
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(kernel_size, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(output_shape[0], activation="softsign")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    print(model.summary())
    return model


# def cos_loss(y_true, y_pred):
#     squared_difference = tf.square(y_true - y_pred)
#     return tf.reduce_mean(squared_difference, axis=-1)


def conv2D(input_shape, output_shape, filters=32, kernel_size=4, dense_size=8):
    max_filters = 1024
    l1_reg = keras.regularizers.l1(l=1e-4)
    l2_reg = keras.regularizers.l2(l=1e-4)
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for add_filters in [16, 32, 64, 128, 256, 512, 1024]:
        # [16,16,32,32,64,64,128,128,256,256,512,512,1024,1024,1024]:  # 10
        ksize = min([x.shape[1], x.shape[2], kernel_size])
        x = layers.SeparableConv2D(
            min(max_filters, add_filters),
            ksize,
            padding="valid",
            bias_initializer=keras.initializers.RandomUniform(),
            bias_regularizer=l1_reg,
            kernel_initializer=keras.initializers.RandomUniform(),
            kernel_regularizer=l1_reg,
        )(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1))(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
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
        # loss=keras.losses.MeanSquaredError(),
        loss=keras.losses.CosineSimilarity(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy", "AUC"],
    )
    print(model.summary())
    return model
