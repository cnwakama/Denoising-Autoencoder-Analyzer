import tensorflow as tf
import numpy as np
import csv
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', '', 'Path to *.npy file')
flags.DEFINE_string('output', '', 'Output file')
flags.DEFINE_string('name', '', 'name of model')
flags.DEFINE_bool('mult_lines', False, 'adding multiple of lines')
flags.DEFINE_string('next_line', '', '')

array = np.load(os.path.expanduser(FLAGS.input))
with open(FLAGS.output, 'a') as fd:
    writer = csv.writer(fd)
    if FLAGS.next_line == 'next_line':
        writer.writerow('')
    if FLAGS.mult_lines:
        for row in array:
            fd.write(FLAGS.name + ',')
            writer.writerow(row)
    else:
        fd.write(FLAGS.name + ',')
        writer.writerow(array)