import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', '', 'Path to dataset.')
flags.DEFINE_string('name', '', 'Name of dataset')
flags.DEFINE_string('directory', '', 'Directory to store information')

data = np.genfromtxt(FLAGS.dataset, skip_header=True, delimiter=',')
data = data[:,1:]

X_train, X_test = train_test_split(data, test_size=0.1)
X_train, X_val= train_test_split(X_train, test_size=0.1)

np.save(os.path.expanduser(FLAGS.directory) + "training_set", X_train)
np.save(os.path.expanduser(FLAGS.directory) + "test_set", X_test)
np.save(os.path.expanduser(FLAGS.directory) + "validation_set", X_val)
