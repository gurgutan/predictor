import tensorflow as tf
from tensorflow import keras

saved_model_dir = "././models/20"
model = keras.models.load_model(saved_model_dir)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tf.lite.TFLiteConverter.from_keras_model
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]

tflite_model = converter.convert()
open("TFLite/20.tflite", "wb").write(tflite_model)
