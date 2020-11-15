from tensorflow.keras.layers import Layer

# from tensorflow.keras.initializers import *
from tensorflow.keras import backend as K
from tensorflow.python.keras.initializers import constant
import tensorflow as tf


class RBFLayer(Layer):
    def __init__(self, units, **kwargs):
        self.output_dim = units
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(input_shape.as_list()[1], self.output_dim),
            initializer="random_uniform",
            trainable=True,
        )
        self.c = self.add_weight(
            name="c", shape=(self.output_dim,), initializer=constant(0), trainable=True
        )
        self.u = self.add_weight(
            name="u",
            shape=(self.output_dim,),
            initializer=constant(1.0),
            trainable=False,
        )
        self.k = self.add_weight(
            name="k",
            shape=(self.output_dim,),
            initializer=constant(1.0),
            trainable=True,
        )
        self.r = self.add_weight(
            name="r",
            shape=(self.output_dim,),
            initializer=constant(1.0),
            trainable=True,
        )
        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        # z = K.pow((K.dot(x, self.w)-self.c), 2)
        z = tf.math.dot(x, self.w)
        kc = self.k * self.c
        kr = self.k * self.r
        a = kc - kr
        b = kc + kr
        kz = self.k * z
        y1 = K.maximum(0.0, kz - a)
        y2 = K.maximum(0.0, -kz + b)
        y = K.minimum(y1, y2)
        y = K.minimum(y, self.u)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        return {
            "w": self.w.numpy(),
            "c": self.c.numpy(),
            "u": self.u.numpy(),
            "k": self.k.numpy(),
            "r": self.r.numpy(),
        }

    def from_config(cls, config):
        return cls(**config)


class RBFLayerExp(Layer):
    def __init__(self, units, **kwargs):
        self.output_dim = units
        super(RBFLayerExp, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(input_shape.as_list()[1], self.output_dim),
            initializer="random_uniform",
            trainable=True,
        )
        self.c = self.add_weight(
            name="c", shape=(self.output_dim,), initializer=constant(0), trainable=True
        )
        # self.k = self.add_weight(
        #     name="k",
        #     shape=(self.output_dim,),
        #     initializer=constant(1.0),
        #     trainable=True,
        # )
        self.sigma = self.add_weight(
            name="sigma",
            shape=(self.output_dim,),
            initializer=constant(1.0),
            trainable=True,
        )
        super(RBFLayerExp, self).build(input_shape)

    def call(self, x):
        z = -K.pow((K.dot(x, self.w) - self.c), 2)
        return K.exp(z * self.sigma * self.sigma)  # * self.k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        return {
            "w": self.w.numpy(),
            "c": self.c.numpy(),
            # "k": self.k.numpy(),
            "sigma": self.sigma.numpy(),
        }

    def from_config(cls, config):
        return cls(**config)
