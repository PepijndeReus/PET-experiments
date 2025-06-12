"""
This file contains the code that is used in main.py to perform machine learning tasks.
"""

import pandas as pd
import numpy as np
import os
import tensorflow

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense

def knn(label, synthesizer=""):
    # load data
    train_data = pd.read_csv(f'data/{label}_train_data{synthesizer}.csv')
    train_labels = pd.read_csv(f'data/{label}_train_labels{synthesizer}.csv')
    val_data = pd.read_csv(f'data/{label}_val_data.csv')
    val_labels = pd.read_csv(f'data/{label}_val_labels.csv')

    # make sure columns are in same order
    val_data = val_data[train_data.columns]

    # make and fit kNN model
    NeighbourModel = KNeighborsClassifier(n_neighbors=2)
    NeighbourModel.fit(train_data, train_labels.values.ravel())

    # make report and add to dataframe
    report = classification_report(val_labels, NeighbourModel.predict(val_data), output_dict=True)

    return report['accuracy']

def lr(label, synthesizer=""):
    # load data
    train_data = pd.read_csv(f'data/{label}_train_data{synthesizer}.csv')
    train_labels = pd.read_csv(f'data/{label}_train_labels{synthesizer}.csv')
    val_data = pd.read_csv(f'data/{label}_val_data.csv')
    val_labels = pd.read_csv(f'data/{label}_val_labels.csv')

    # make sure columns are in same order
    val_data = val_data[train_data.columns]

    # make and fit logistic regression model
    LogReg = LogisticRegression(max_iter=1000)
    LogReg.fit(train_data, train_labels.values.ravel())

    # make report and add to dataframe
    report = classification_report(val_labels, LogReg.predict(val_data), output_dict=True)

    return report['accuracy']

def nn(label, synthesizer=""):
    train_data = pd.read_csv(f'data/{label}_train_data{synthesizer}.csv')
    train_labels = pd.read_csv(f'data/{label}_train_labels{synthesizer}.csv')
    val_data = pd.read_csv(f'data/{label}_val_data.csv')
    val_labels = pd.read_csv(f'data/{label}_val_labels.csv')

    # make sure columns are in same order
    val_data = val_data[train_data.columns]

    # Draft simple neural network using keras
    NeuralNet = Sequential()
    NeuralNet.add(Dense(32, activation = 'relu'))
    NeuralNet.add(Dense(64, activation = 'relu'))
    NeuralNet.add(Dense(32, activation = 'relu'))
    NeuralNet.add(Dense(8, activation = 'relu'))
    NeuralNet.add(Dense(1, activation = 'sigmoid'))
    NeuralNet.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Failed to convert a NumPy array to a Tensor (Unsupported object type float).
    binary_columns = train_data.columns[(train_data == False).all() | (train_data == True).all()]
    # Convert binary columns to boolean
    train_data[binary_columns] = train_data[binary_columns].astype(int)
    
    # train network
    NeuralNet.fit(train_data, train_labels, batch_size = 150, epochs = 50, verbose=0)
    
    # Failed to convert a NumPy array to a Tensor (Unsupported object type bool).
    # Fix: convert to Tensor
    val_data = tensorflow.convert_to_tensor(val_data, dtype=tensorflow.float32)
    val_labels = tensorflow.convert_to_tensor(val_labels, dtype=tensorflow.int32)

    loss, accuracy = NeuralNet.evaluate(val_data, val_labels, verbose=0)

    return accuracy


if __name__ == "__main__":
    label = str(input("Which dataset would you like to synthesise?\nPick: breast, census, heart or student.")).lower()
    synthesizer = str(input("Which synthetic data generator would you like to synthesise?\nPick: benchmark, dpctgan, ydata, ctgan, synthcity, nbsynthetic or datasynthesizer.")).lower()
    input_ml = str(input("Which machine learning technique would you like to use?\nPick: knn, lr or nn.\nNote: use abbreviations. knn = k-nearest neighbours, lr = logistic regression and nn = neural network.")).lower()

    # pick right method
    if input_ml == 'knn':
        knn(label, synthesizer)
    elif input_ml == 'lr':
        lr(label, synthesizer)
    elif input_ml == 'nn':
        nn(label, synthesizer)
    else:
        print("The machine learning method you entered is invalid. Please try again")
        raise KeyError('method not in list of machine learning techniques.')
