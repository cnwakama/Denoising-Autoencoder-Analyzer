import numpy as np
import tensorflow as tf
import pandas as pd
import evalidate as e
import seaborn as sns
import matplotlib.pyplot as plt
import os

from os import listdir
from os.path import isfile, join

def create_labels(label_path, training_path, name):
    data = np.genfromtxt(training_path, delimiter=',', skip_header=1)
    id = np.genfromtxt(training_path, delimiter=',', usecols=[0], dtype=str, skip_header=1)
    df = pd.read_csv(label_path, sep="\t", skiprows=0, usecols=[1, 7], dtype=str)
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

    np.save(name, data)
    print("Complete")

def compressed_inputs(directory_path, dataset_path, output_directory):
    # used to load a csv file
    # data = np.genfromtxt(dataset_path, skip_header=1, delimiter=',')
    # data = data[:, 1:]

    # load a .npy file
    data = np.load(dataset_path)

    normalized_data = tf.keras.utils.normalize(data)

    evalidate = e.Evalidate()

    for file in os.listdir(directory_path):
        if file.endswith(".meta"):
            encoded_data = evalidate.encoder_tensor(normalized_data, os.path.join(directory_path, file.split('.')[0]))
            print (np.shape(encoded_data))
            np.savetxt(os.path.join(output_directory, os.path.splitext(file)[0]+".csv"), encoded_data, delimiter=",")
            print ("Model: " + os.path.splitext(file)[0])

    print ("Complete")

def load_dataset(dataset="dataset.npy", input_dim=20531):
    # load dataset holding features and label (at the last column)
    dataset = np.load(dataset)

    # split into input (X) and output (Y) variables
    X = dataset[:, 0:input_dim]
    Y = dataset[:, input_dim]

    return X,Y

def create_variables(self, data_path):
    dataset = np.load(data_path)
    length = np.shape(dataset)[1] - 1
    #split = dataset.shape[1] - 1
    training = dataset[:, 0:length]
    labels = dataset[:, length]
    #training, labels = np.hsplit(dataset, [split])

    return training, labels

def generate_distribution(dataset, col=["feature" + str(a) for a in range(20531)], var_names=[], melt=True, show=False):
    if len(var_names) == 0:
        search = col
    else:
        search = var_names
    data = np.genfromtxt(dataset, delimiter=',')
    data_norm = tf.keras.utils.normalize(data, axis=1)

    df = pd.DataFrame(data=data_norm, columns=col)
    df_filter = df[search]

    _, name = os.path.split(dataset)
    name = os.path.splitext(name)[0]
    variables = ''.join(['_' + x for x in search])
    name = name + variables + ".png"

    if melt:
        # melt method
        dfm = pd.melt(df_filter, var_name='columns')
        g = sns.FacetGrid(dfm, col='columns')
        g.map(sns.distplot, 'value')
        g.savefig(os.path.splitext(name)[0])
    else:
        # loop method
        fig, axes = plt.subplots(ncols=len(var_names))
        for ax, col in zip(axes, df_filter.columns):
            sns.distplot(df_filter[col], ax=ax)
            # plt.savefig(name + '_' + str(col) + '.png')
        plt.savefig(name)

    if show:
        plt.show()
