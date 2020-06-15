import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
# from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, BatchNormalization, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM, Reshape


def conv2D_aka_inception(input_shape,
                         output_shape,
                         filters=32,
                         kernel_size=4,
                         dense_size=8):
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
        residual = layers.Conv2D(size, 1, strides=2,
                                 padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(output_shape[0], activation="softsign")(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    print(model.summary())
    return model


def conv2D(input_shape, output_shape, filters=32, kernel_size=4, dense_size=8):
    l1_reg = keras.regularizers.l1(l=1e-4)
    l2_reg = keras.regularizers.l2(l=1e-4)
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=1,
                      bias_initializer=keras.initializers.RandomNormal(),
                      bias_regularizer=l2_reg,
                      kernel_initializer=keras.initializers.RandomNormal(),
                      kernel_regularizer=l2_reg,
                      padding="valid")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for i in range(4):
        x = layers.Conv2D(
            min(256, filters * 2**i),
            kernel_size,  #max(1, kernel_size - i),
            padding="valid",
            bias_initializer=keras.initializers.RandomNormal(),
            bias_regularizer=l2_reg,
            kernel_initializer=keras.initializers.RandomNormal(),
            kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_size,
                     activation="softsign",
                     bias_initializer=keras.initializers.RandomNormal(),
                     bias_regularizer=l2_reg,
                     kernel_initializer=keras.initializers.RandomNormal(),
                     kernel_regularizer=l2_reg)(x)

    outputs = layers.Dense(
        output_shape[0],
        # activation="softsign",
        activation="softmax",
        bias_initializer=keras.initializers.RandomNormal(),
        bias_regularizer=l1_reg,
        kernel_initializer=keras.initializers.RandomNormal(),
        kernel_regularizer=l1_reg)(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        # loss=keras.losses.MeanSquaredError(),
        loss=keras.losses.KLDivergence(),
        optimizer=keras.optimizers.SGD(1e-2),  #, nesterov=True),
        metrics=['accuracy'])
    print(model.summary())
    return model
