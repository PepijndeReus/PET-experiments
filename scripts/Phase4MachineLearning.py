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

	# name = 'benchmark' if synthesizer=="" else synthesizer.replace('_','')
	# df = pd.DataFrame({'measurement': [name], 'accuracy knn': [report['accuracy']]})
	# df.to_csv(f'measurements/accuracy_{label.split("/")[-1]}.csv',
	# 			header=not os.path.exists(f'measurements/accuracy_{label.split("/")[-1]}.csv'),
	# 			index=False, mode='a')

	#df = pd.DataFrame([report['accuracy']])
	#df.columns = ['accuracy knn', 'accuracy lr', 'accuracy nn']
	#df.to_csv(f'measurements/accuracy_{label.split("/")[-1]}.csv',
#				header=not os.path.exists(f'measurements/accuracy_{label.split("/")[-1]}.csv'),
#				index=False, mode='a')

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

	# name = 'benchmark' if synthesizer=="" else synthesizer.replace('_','')
	# df = pd.DataFrame({'measurement': [name], 'accuracy lr': [report['accuracy']]})
	# df.to_csv(f'measurements/accuracy_{label.split("/")[-1]}.csv',
	# 			header=not os.path.exists(f'measurements/accuracy_{label.split("/")[-1]}.csv'),
	# 			index=False, mode='a')

	return report['accuracy']
	#df.columns = ['Accuracy']
	#df.to_csv(f'measurements/accuracy_{label.replace("/","_")}{synthesizer}_logreg.csv', index=False)

def nn(label, synthesizer=""):
	train_data = pd.read_csv(f'data/{label}_train_data{synthesizer}.csv')
	train_labels = pd.read_csv(f'data/{label}_train_labels{synthesizer}.csv')
	val_data = pd.read_csv(f'data/{label}_val_data.csv')
	val_labels = pd.read_csv(f'data/{label}_val_labels.csv')

	# make sure columns are in same order
	val_data = val_data[train_data.columns]

	# Regel hieronder: tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
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
	
	NeuralNet.fit(train_data, train_labels, batch_size = 150, epochs = 50, verbose=0)
	
	# Failed to convert a NumPy array to a Tensor (Unsupported object type bool).
	# Fix: convert to Tensor
	val_data = tensorflow.convert_to_tensor(val_data, dtype=tensorflow.float32)
	val_labels = tensorflow.convert_to_tensor(val_labels, dtype=tensorflow.int32) 

	loss, accuracy = NeuralNet.evaluate(val_data, val_labels, verbose=0)

	# name = 'benchmark' if synthesizer=="" else synthesizer.replace('_','')
	# df = pd.DataFrame({'measurement': [name], 'accuracy nn': accuracy})
	# df.to_csv(f'measurements/accuracy_{label.split("/")[-1]}.csv',
	# 			header=not os.path.exists(f'measurements/accuracy_{label.split("/")[-1]}.csv'),
	# 			index=False, mode='a')x

	return accuracy


if __name__ == "__main__":
	knn('syn/breast', '_synthcity')
	exit()
	lr('breast')
	exit()
	knn('student')

	logreg('breast')
	logreg('student')

	nn('breast')
	nn('student')
