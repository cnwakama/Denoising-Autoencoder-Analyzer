import tensorflow as tf
from vis.visualization import visualize_activation
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from vis.utils import utils


class Evalidate():

    def __init__(self):
        # tensorflow objects
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.model = None
        self.model_path = None

        # training parameters
        self.batch_size = 128
        self.nb_epoch = 20
        self.input_dim = 20531
        self.l1 = 0.01
        self.iter = 500

    def get_tensor_model(self):
        with self.tf_graph.as_default():
            saver = tf.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.tf_session, self.model_path)

    def weight_init(self, shape, dtype=None, partition_info=None):
        weight = self.tf_session.run(self.tf_graph.get_tensor_by_name("enc-w:0"))
        weight.shape = (shape)
        return weight

    def bias_init(self, shape, dtype=None, partition_info=None):
        bias = self.tf_session.run(self.tf_graph.get_tensor_by_name("hidden-bias:0"))
        bias.shape = shape
        return bias

    def build_model(self, n, model_path, activation="linear"):
        # Get TensorFlow model path
        self.model_path = model_path
        self.get_tensor_model()

        input = tf.keras.layers.Input(shape=(n,), name="input")

        # Layers
        encoded_input = tf.keras.layers.Dense(100, kernel_initializer=self.weight_init, bias_initializer=self.bias_init,
                                              kernel_regularizer=tf.keras.regularizers.l2(self.l1))(input)
        predictions = tf.keras.layers.Dense(1, activation=activation)(encoded_input)

        model = tf.keras.models.Model(input, predictions)
        model.layers[1].trainable = False

        model.compile(loss="binary_crossentropy", optimizer="sgd",
                      metrics=["binary_accuracy", "mae", "mse"])
        model.summary()

    def train_model(self, data_path, val=0, test_X=None, test_Y=None):
        train_X, train_Y = self._create_variables(data_path)
        history = self.model.fit(train_X, train_Y, verbose=1, validation_split=val)

        # Metric Analysis
        predictions = history.predict(train_X)
        rounded = [round(x[0]) for x in predictions]

        score = history.evaluate(train_X, train_Y, verbose=0)

        print('Training score:', score[0])
        print('Training accuracy:', score[1])
        print('Training accuracy:', score[2])
        print('Training accuracy:', score[3])
        print('Prediction:', predictions)
        print('Estimated Prediction:', rounded)

        if test_X != None:
            score = history.evaluate(test_X, test_Y, verbose=0)

            print('Test score:', score[0])
            print('Test accuracy:', score[1])
            print('Test accuracy:', score[2])
            print('Test accuracy:', score[3])

    def min_death(self, data_path, model_path):
        train_X, train_Y = self._create_variables(data_path)
        self.build_model(train_X.shape[0], model_path)
        self.train_model(train_X, train_Y)
        feature_extract = visualize_activation(self.model, layer_idx=1, filter_indices=None, grad_modifier="negate",
                                               input_range=(0, 1), seed_input=1, max_iter=self.iter, verbose=False)

        print("Shape of Extract:", feature_extract.shape)
        print(feature_extract)

    def create_labels(self, label_path, training_path):
        labels = np.genfromtxt(label_path, dtype=None, delimiter=',', skip_header=True)
        data = np.genfromtxt(training_path, dtype=None, delimiter=',', skip_header=True)
        z = np.zeros((data.shape[0], 1))
        z = ('vital_status', z)
        np.append(data, z, axis=1)

        columns = data.shape[1]
        for x in range(len(z)):
            for y in range(labels.shape[0]):
                if labels[x, 1] == data[y, 0]:
                    data[y, columns - 1] = 1 if labels[x, 7] == 'alive' else 0
                    continue

        np.save("dataset", data)
        print("Complete")

    def _create_variables(self, data_path):
        dataset = np.load(data_path)
        training, labels = np.hsplit(dataset, dataset.shape[1] - 1)

        return training, labels
