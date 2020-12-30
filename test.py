import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.engine.base_layer import Layer
from predictor import Predictor
import matplotlib.pyplot as plt
import pandas as pd
import math
from models import esum2, esum
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Concatenate
import pydot


def cross(kernels):
    return lambda x: Concatenate()([l(x) for l in kernels])


def add(kernels):
    return (
        lambda x: kernels[0](x)
        if (len(kernels) == 1)
        else kernels[-1](add(kernels[:-1])(x))
    )


i = Input(shape=(8,))
u1 = [Dense(8), Dense(16), Dense(32)]
u2 = [Dense(3), Dense(3)]
u3 = [Dense(5), add(u2)]
a = add([Dense(8), cross([Dense(10), add(u2), cross(u3)])])
model = tf.keras.Model(inputs=i, outputs=a(i))
# model = tf.keras.Model(inputs=i, outputs=cross(u1)(i))
tf.keras.utils.plot_model(
    model, show_shapes=True, show_layer_names=False, to_file="models/test1.png"
)

