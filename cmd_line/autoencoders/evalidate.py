import tensorflow as tf
from vis.visualization import visualize_activation
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from vis.utils import utils


class Evalidate():

    def __init__(self):
        # tensorflow objects
        self.tf_graph = tf.get_default_graph()
        self.tf_session = tf.Session()
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

    def _weight_init(self, shape, dtype=None, partition_info=None):
        weight = self.tf_session.run(self.tf_graph.get_tensor_by_name("enc-w:0"))
        weight.shape = (shape)
        return weight

    def _bias_init(self, shape, dtype=None, partition_info=None):
        bias = self.tf_session.run(self.tf_graph.get_tensor_by_name("hidden-bias:0"))
        bias.shape = shape
        return bias

    def _coeff_determination(self, y_true, y_pred):
        K = tf.keras.backend
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res/(SS_tot + K.epsilon())

    def build_model(self, model_path, n=20531, activation="linear"):
        # Get TensorFlow model path
        self.model_path = model_path
        self.get_tensor_model()

        input = tf.keras.layers.Input(shape=(n,), name="input")

        # Layers
        encoded_input = tf.keras.layers.Dense(100, kernel_initializer=self._weight_init, bias_initializer=self._bias_init,
                                              kernel_regularizer=tf.keras.regularizers.l2(self.l1))(input)
        encoded_input = tf.keras.layers.BatchNormalization()(encoded_input)
        predictions = tf.keras.layers.Dense(1, activation=activation)(encoded_input)

        self.model = tf.keras.models.Model(input, predictions)
        self.model.layers[1].trainable = False

        self.model.compile(loss="mean_squared_error", optimizer="sgd",
                      metrics=["accuracy", "mae", "mse", self._coeff_determination])
        self.model.summary()


    def train_model(self, data_path, val=0, test_X=None, test_Y=None):
        train_X, train_Y = self._create_variables(data_path)

        # normalized
        trX = tf.keras.utils.normalize(train_X)
        regression = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=self.model, epochs=100, batch_size=5, verbose=1)
        history = self.model.fit(train_X, train_Y, verbose=1, validation_split=val)

        kFold = KFold(n_splits=10, random_state=1)
        results = cross_val_score(regression, train_X, train_Y, cv=kFold)
        results.mean()
        results.std()


        # Metric Analysis
        predictions = self.model.predict(train_X)
        rounded = [round(x[0]) for x in predictions]

        score = self.model.evaluate(train_X, train_Y, verbose=0)

        print('Training score:', score[0])
        print('Training accuracy:', score[1])
        print('Mean Absolute Error:', score[2])
        print('Mean Squared Error:', score[3])
        print('Prediction:', predictions)
        print('Estimated Prediction:', rounded)

        if test_X != None:
            score = self.model.evaluate(test_X, test_Y, verbose=0)

            print('Test score:', score[0])
            print('Test accuracy:', score[1])
            print('Test accuracy:', score[2])
            print('Test accuracy:', score[3])

        return trX

    def min_death(self, data_path, model_path):
        train_X, train_Y = self._create_variables(data_path)
        self.build_model(train_X.shape[0], model_path)
        normalX = self.train_model(train_X, train_Y)
        feature_extract = visualize_activation(self.model, layer_idx=1, filter_indices=None, grad_modifier="negate",
                                               input_range=(np.min(normalX), np.max(normalX)), seed_input=1, max_iter=self.iter, verbose=False)

        print("Shape of Extract:", feature_extract.shape)
        print(feature_extract)

    def create_labels(self, label_path, training_path):
        data = np.genfromtxt(training_path, delimiter=',', skip_header=1)
        id = np.genfromtxt(training_path, delimiter=',', usecols=[0], dtype=str, skip_header=1)
        df = pd.read_csv(label_path, sep="\t", skiprows=0, usecols=[1,7], dtype=str)
        labels = df.values
        z = np.zeros((data.shape[0], 1))
        data = data[:, 1:data.shape[1]]
        data = np.append(data, z, 1)
        columns = data.shape[1]

        for x in range(labels.shape[0]):
            for y in range(labels.shape[0]):
                if str(labels[x, 0]) == str(id[y]).replace('"', ''):
                    data[y, columns - 1] = 1 if str(labels[x, 1]) == 'alive' else 0
                    # print(str(labels[x, 1]) + ":", data[y, columns - 1])
                    continue


        np.save("dataset", data)
        print("Complete")

    def _create_variables(self, data_path):
        dataset = np.load(data_path)
        split = dataset.shape[1] - 1
        training, labels = np.hsplit(dataset, [split])

        return training, labels

e = Evalidate()
e.build_model("/Users/chibuzonwakama/yadlt/models/dae_model47")
#e.train_model("dataset.npy")
# e.create_labels("clinical.tsv", "rna_solidtumor_tcgahnsc.csv")
