"""
Text
"""

import pandas as pd
import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense


def measure_knn(label):
	# load data
	train_data = pd.read_csv(f'data/{label}_train_data.csv')
	train_labels = pd.read_csv(f'data/{label}_train_labels.csv')
	val_data = pd.read_csv(f'data/{label}_val_data.csv')
	val_labels = pd.read_csv(f'data/{label}_val_labels.csv')

	# make and fit kNN model
	NeighbourModel = KNeighborsClassifier(n_neighbors=5) # Todo, is dit niet twee?
	NeighbourModel.fit(train_data, train_labels.values.ravel())

	# make report and add to dataframe
	report = classification_report(val_labels, NeighbourModel.predict(val_data), output_dict=True)
	df = pd.DataFrame([report['accuracy']])
	df.columns = ['Accuracy']
	df.to_csv(f'measurements/accuracy_{label}_knn.csv', index=False)

def measure_logreg(label):
	# load data
	train_data = pd.read_csv(f'data/{label}_train_data.csv')
	train_labels = pd.read_csv(f'data/{label}_train_labels.csv')
	val_data = pd.read_csv(f'data/{label}_val_data.csv')
	val_labels = pd.read_csv(f'data/{label}_val_labels.csv')

	# make and fit logistic regression model
	LogReg = LogisticRegression(max_iter=1000)
	LogReg.fit(train_data, train_labels.values.ravel())

	# make report and add to dataframe
	report = classification_report(val_labels, LogReg.predict(val_data), output_dict=True)
	df = pd.DataFrame([report['accuracy']])
	df.columns = ['Accuracy']
	df.to_csv(f'measurements/accuracy_{label}_logreg.csv', index=False)

def measure_nn(label):
	train_data = pd.read_csv(f'data/{label}_train_data.csv')
	train_labels = pd.read_csv(f'data/{label}_train_labels.csv')
	val_data = pd.read_csv(f'data/{label}_val_data.csv')
	val_labels = pd.read_csv(f'data/{label}_val_labels.csv')

	NeuralNet = Sequential()
	NeuralNet.add(Dense(8, activation = 'relu'))
	NeuralNet.add(Dense(16, activation = 'relu'))
	NeuralNet.add(Dense(1, activation = 'sigmoid'))
	NeuralNet.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

	NeuralNet.fit(train_data, train_labels, batch_size = 150, epochs = 20, verbose=0)

	loss, accuracy = NeuralNet.evaluate(val_data, val_labels, verbose=0)
	df = pd.DataFrame([[accuracy]])
	df.columns = ['Accuracy']
	df.to_csv('measurements/accuracy_{label}_nn.csv', index=False)

if __name__ == "__main__":
	measure_knn('breast')
	measure_knn('student')

	measure_logreg('breast')
	measure_logreg('student')

	measure_nn('breast')
	measure_nn('student')
