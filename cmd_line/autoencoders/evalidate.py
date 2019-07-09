import tensorflow as tf
import numpy as np
import statsmodels.api as sm
import utilities as util


from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_classif, f_regression, f_oneway
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from scipy import stats
from vis.visualization import visualize_activation

# global variable
model_path = "/Volumes/Files Backups/Document_12-12-18/New Folder With Items/yadlt/models/dae_model47"

class Evalidate():

    def __init__(self, X, Y):
        # training parameters
        self.seed = 1
        self.batch_size = 5
        self.nb_epoch = 5
        self.l1 = 0.01
        self.iter = 500
        self.verbose = 1
        self.cv = 3

        self.input_dim = np.shape(X)[1]
        self.X = X
        self.Y = Y

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

    def basic_analysis(self):
        # getting pval using two different methods
        f1, p1 = f_regression(self.X, self.Y)
        f2, p2, = f_classif(self.X, self.Y)

        return p1, p2, f1, f2

    # use stat module to print a summary of logistic and linear regression
    def regression_stat(self):
        print ("Linear Regression")
        x2 = sm.add_constant(self.X)
        est = sm.OLS(self.Y, x2)
        est2 = est.fit()
        print(est2.summary())
        print(est2.summary2())

        print ("Logistics Regression")
        logit_model = sm.Logit(self.Y, self.X)
        result = logit_model.fit()
        print(result.summary())
        print(result.summary2())

    # calculate pval with matrix algebra
    def get_pval(self, Y_hat, p, n):
        vector_Y = np.ones((1, np.shape(Y_hat)[1]))
        mean_Y = np.mean(self.Y)
        vector_Y.fill(mean_Y[0])
        msr = np.sum(np.square(Y_hat - vector_Y))/p
        mse = np.sum(np.square(self.Y - Y_hat))/(n - p - 1)

        f_val = msr/mse
        # p_val =



    def train_model(self, model_path, input_dim=20531, normalized=True):
        # fix random seed for reproducibility
        seed = 1
        np.random.seed(seed)

        if normalized:
            X = tf.keras.utils.normalize(self.X)
        else:
            X = self.X

        # create model
        model = KerasRegressor(build_fn=self.build_model, epochs=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose)

        # compress data
        X_compressed = self.encoder_tensor(X, model_path=model_path)

        print (X_compressed)
        print (np.shape(X_compressed))

        results = model.fit(X_compressed, self.Y)


        # evaluate using n-fold cross validation
        scoring = ['roc_auc', 'balanced_accuracy', 'r2', 'neg_mean_squared_error']
        kfold = KFold(n_splits=self.cv, random_state=seed)
        results = cross_validate(model, X_compressed, self.Y, cv=kfold, verbose=1)

        print ("Results")
        print ("ROC: ", results['test_roc_auc'])
        print ("Accuracy: ", results['test_balanced_accuracy'])
        print ("R2: ", results['test_r2'])
        print ("Mean Squared Error: ", results['test_neg_mean_squared_error'])



    def min_death(self, data_path, model_path, model, normalized=True):
        #train_X, train_Y = self._create_variables(data_path)
        #self.build_model(self.X.shape[0], model_path)
        # normalX = self.train_model(self.X, self.Y)
        if normalized:
            normalX = tf.keras.utils.normalize(self.X)
        else:
            normalX = self.X
        feature_extract = visualize_activation(model, layer_idx=1, filter_indices=None, grad_modifier="negate",
                                               input_range=(np.min(normalX), np.max(normalX)), seed_input=self.seed,
                                               max_iter=self.iter, verbose=self.verbose)

        print("Shape of Extract:", feature_extract.shape)
        print(feature_extract)





# X, Y = util.load_dataset(input_dim=self.input_dim)
# test prodecure
# eval = Evalidate()
# model = eval.build_model()
# print (model.summary())
# eval.train_model(model_path=model_path)
