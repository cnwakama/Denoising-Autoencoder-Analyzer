import tensorflow as tf
import numpy as np
import os
import glob

flags = tf.app.flags
FLAGS = flags.FLAGS

# parameter
# flags.DEFINE_string('model', '', 'The path to *.meta file for model')
flags.DEFINE_string('directory', '', 'Directory of .meta files')

'''
Gets the weights from the .meta file and saves weights in to csv file
'''
def get_weights(model):
        # resetting enviroment graph
        tf.reset_default_graph()

        with tf.Session() as sess:
                with tf.get_default_graph().as_default() as graph:
                        # configures and restores graph from trained model  
                        saver = tf.train.import_meta_graph(model)
                        saver.restore(sess, os.path.splitext(model)[0])

                        # get tensor of weights 
                        weights = graph.get_tensor_by_name('enc-w:0')

                        # evalulate tensor return a numpy array of trained weights in the model
                        weight_matrix = weights.eval()

                        # saves weights into a csv file
                        np.savetxt(os.path.basename(model).split('.')[0] + ".csv", weight_matrix, delimiter=",")



if __name__ == '__main__':
        for file in glob.glob(FLAGS.directory + "/*.meta"):
		get_weights(file)

