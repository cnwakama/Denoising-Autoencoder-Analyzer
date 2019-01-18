import utilities as u
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', '', 'Path to dataset.')
flags.DEFINE_string('directory', '', 'Path to directory holding model information')
flags.DEFINE_string('output', '', 'output directory path')

u.compressed_inputs(FLAGS.directory, FLAGS.dataset, FLAGS.output)