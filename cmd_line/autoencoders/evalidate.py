import tensorflow as tf
import numpy as np


from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_classif, f_regression, f_oneway
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
# from vis.visualization import visualize_activation

# global variable
model_path = "/Volumes/Files Backups/Document_12-12-18/New Folder With Items/yadlt/models/dae_model47"

class Evalidate():

    def __init__(self):
        # training parameters
        self.batch_size = 5
        self.nb_epoch = 5
        self.input_dim = 20531
        self.l1 = 0.01
        self.iter = 500
        self.verbose = 1
        self.cv = 3
        self.X = None
        self.Y = None

    def _coeff_determination(self, y_true, y_pred):
        K = tf.keras.backend
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res / (SS_tot + K.epsilon())

    def encoder_tensor(self, data, model_path):
        tf.reset_default_graph()
        print (model_path)
        with tf.Session() as sess:
            with tf.get_default_graph().as_default() as graph:
                saver = tf.train.import_meta_graph(model_path + '.meta')
                saver.restore(sess, model_path)

            name_scope = 'encoder'
            encoded_tensor = self.get_model_activation_func(name_scope, sess.graph)
            input = graph.get_tensor_by_name('x-corr-input:0')

            encoded_data = encoded_tensor.eval({input: data})

            return encoded_data

    def get_model_activation_func(self, name_scope, graph):
        activation = ['Sigmoid', 'Relu', 'Tanh']
        activation_scope = [name_scope + '/' + a for a in activation]
        ops = graph.get_operations()

        pos = ''
        for item in activation_scope:
            if item in [i.name for i in ops]:
                pos = item
                break

        encoder = graph.get_tensor_by_name(pos + ':0')

        return encoder


    def build_model(self, activation = "linear", n_feature=100):

        input = Input(shape=(n_feature,))
        output = Dense(1, activation=activation)(input)

        model = Model(input, output)
        model.compile(loss='mse', optimizer='sgd', metrics=['accuracy', 'mae'])

        return model

    # using
    def basic_analysis(self):
        f1, p1 = f_regression(self.X, self.Y)
        f2, p2, = f_classif(self.X, self.Y)

    # use stat module
    #def regression_stat(self):

    # calculate pval with matrix algebra
    #def get_pval(self):



    def train_model(self, model_path, input_dim=20531, normalized=True):
        # fix random seed for reproducibility
        seed = 1
        np.random.seed(seed)

        # load dataset
        dataset = np.load("dataset.npy")

        # split into input (X) and output (Y) variables
        X = dataset[:, 0:input_dim]
        Y = dataset[:, input_dim]

        if normalized:
            X = tf.keras.utils.normalize(X)

        # create model
        model = KerasRegressor(build_fn=self.build_model, epochs=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose)

        # compress data
        X_compressed = self.encoder_tensor(X, model_path=model_path)

        print (X_compressed)
        print (np.shape(X_compressed))

        results = model.fit(X_compressed, Y)


        # # evaluate using n-fold cross validation
        # scoring = ['roc_auc', 'accuracy', 'r2']
        # kfold = KFold(n_splits=self.cv, random_state=seed)
        # results = cross_val_score(model, X_compressed, Y, cv=kfold, verbose=1)
        # print(results.mean())
        # print(results.std())
        #
        # # from previous of evalidate
        # # history = self.model.fit(trX, train_Y, verbose=1, validation_split=val)
        #
        # # {'r2', 'roc_auc', 'balanced_accuracy', 'roc_curve', 'neg_mean_squared_error', 'neg_mean_absolute_error'}
        # # kFold = KFold(n_splits=10, random_state=1)
        # # results = cross_val_score(regression, trX, train_Y)
        #
        # # print (np.shape(results))
        # # print (np.shape(history))
        # # proba = cross_val_predict(regression, train_X, train_Y, cv=kFold, method='predict_proba')
        # # results.mean()
        # # results.std()
        #
        # r2_sc, perm_sc, pval = permutation_test_score(regression, trX, train_Y, n_permutations=100, scoring='r2',
        #                                               cv=None)
        # proba = self.model.predict_proba(train_X)
        # print (proba)
        # print (r2_sc)
        #
        # # Metric Analysis
        # predictions = self.model.predict(train_X)
        # rounded = [round(x[0]) for x in predictions]
        #
        # score = self.model.evaluate(train_X, train_Y, verbose=0)
        #
        # print('Training score:', score[0])
        # print('Training accuracy:', score[1])
        # print('Mean Absolute Error:', score[2])
        # print('Mean Squared Error:', score[3])
        # print('Prediction:', predictions)
        # print('Estimated Prediction:', rounded)
        #
        # if test_X != None:
        #     score = self.model.evaluate(test_X, test_Y, verbose=0)
        #
        #     print('Test score:', score[0])
        #     print('Test accuracy:', score[1])
        #     print('Test accuracy:', score[2])
        #     print('Test accuracy:', score[3])
        #
        # return trX

    def min_death(self, data_path, model_path):
        train_X, train_Y = self._create_variables(data_path)
        self.build_model(train_X.shape[0], model_path)
        normalX = self.train_model(train_X, train_Y)
        # feature_extract = visualize_activation(self.model, layer_idx=1, filter_indices=None, grad_modifier="negate",
        #                                        input_range=(np.min(normalX), np.max(normalX)), seed_input=1,
        #                                        max_iter=self.iter, verbose=False)
        #
        # print("Shape of Extract:", feature_extract.shape)
        # print(feature_extract)



    def _create_variables(self, data_path):
        dataset = np.load(data_path)
        length = np.shape(dataset)[1] - 1
        #split = dataset.shape[1] - 1
        training = dataset[:, 0:length]
        labels = dataset[:, length]
        #training, labels = np.hsplit(dataset, [split])

        return training, labels


# test prodecure
# eval = Evalidate()
# model = eval.build_model()
# print (model.summary())
# eval.train_model(model_path=model_path)
