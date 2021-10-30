import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.backend import random_binomial
from tensorflow.python.keras.engine.base_layer import Layer
from predictor import Predictor
import matplotlib.pyplot as plt
import pandas as pd
import math
from models import esum2, esum
import random
from tensorflow.keras.layers import Dense, Activation, Input, Concatenate
import pydot


def add(kernels):
    return (
        lambda x: kernels[0](x)
        if (len(kernels) == 1)
        else Concatenate()([l(x) for l in kernels])
    )


def mul(kernels):
    return (
        lambda x: kernels[0](x)
        if (len(kernels) == 1)
        else kernels[-1](mul(kernels[:-1])(x))
    )


def random_net(depth):
    if depth == 0:
        return Dense(1)
    r = random.randint(0, 5)
    if r == 0:
        return mul([random_net(depth - 1), random_net(depth - 1)])
    elif r == 1:
        return mul([Dense(1), random_net(depth - 1)])
    elif r == 2:
        return mul([random_net(depth - 1), Dense(1)])
    elif r == 3:
        return add([Dense(1), random_net(depth - 1)])
    elif r == 4:
        return add([random_net(depth - 1), Dense(1)])
    elif r == 5:
        return add([random_net(depth - 1), random_net(depth - 1)])


net = random_net(8)
i = Input(shape=(1,), name="i")
o = Dense(1, name="o")
# d1 = Dense(1, name="d1")
# d2 = Dense(1, name="d2")
# d3 = Dense(1, name="d3")
# d4 = Dense(1, name="d4")
# d5 = Dense(1, name="d5")
# d6d7 = [Dense(1, name="d6"), Dense(1, name="d7")]
# a = add([d1, cross([add([cross([d2, d5]), cross(d6d7)]), cross([d3, d4])]), o])
model = tf.keras.Model(inputs=i, outputs=net(i))
# model = tf.keras.Model(inputs=i, outputs=cross(u1)(i))
tf.keras.utils.plot_model(
    model,
    show_shapes=False,
    rankdir="LR",
    show_layer_names=True,
    to_file="models/test1.png",
)

