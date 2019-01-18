import numpy as np
import tensorflow as tf
import pandas as pd
import evalidate as e
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
    data = np.load(dataset_path)

    normalized_data = tf.keras.utils.normalize(data)

    evalidate = e.Evalidate()

    for file in os.listdir(directory_path):
        if file.endswith(".meta"):
            encoded_data = evalidate.encoder_tensor(normalized_data, file)
            np.savetxt(os.path.join(output_directory, os.path.splitext(file)[0]+".csv"), encoded_data, delimiter=",")
            print ("Model: " + os.path.splitext(file)[0])

    print ("Complete")