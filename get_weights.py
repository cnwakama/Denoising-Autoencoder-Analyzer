import tensorflow as tf
import numpy as np
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', '', 'The path to *.meta file for model')


# model_path = 'dae_model97'


def get_weights(model):
        tf.reset_default_graph()

        with tf.Session() as sess:
                with tf.get_default_graph().as_default() as graph:
                        saver = tf.train.import_meta_graph(model)
                        saver.restore(sess, os.path.splitext(model)[0])

                        weights = graph.get_tensor_by_name('enc-w:0')

                        weight_matrix = weights.eval()
                        np.savetxt(os.path.basename(model).split('.')[0] + ".csv", weight_matrix, delimiter=",")



if __name__ == '__main__':
        get_weights(FLAGS.model)

