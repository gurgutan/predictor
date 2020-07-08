import tensorflow as tf
from tensorflow.python.framework import graph_io

frozen = tf.graph_util.convert_variables_to_constants(
    sess, sess.graph_def, ["name_of_the_output_node"]
)
graph_io.write_graph(frozen, "./", "inference_graph.pb", as_text=False)

