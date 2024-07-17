"""
This python script will pepare the (synthetic) data for the machine learning
task.

Please note that this data has already been cleaned (see 1_cleaning.py). Data
should be present in the 'data/' folder.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preproc_breast(path=""):
	'''Preprocesses breast-cancer dataset '''
	breast_train = pd.read_csv(f'data/{path}breast_train.csv')
	breast_val = pd.read_csv(f'data/{path}breast_val.csv')

	def preprocess(data):
		# make binary labels for recurrence
		data['class'] = data['class'].str.replace('no-recurrence-events', '0')
		data['class'] = data['class'].str.replace('recurrence-events', '1')
		data['class'] = data['class'].astype(int)

		# split data with labels
		labels = data['class'].copy()
		data = data.drop(['class'], axis=1)

		# use One-hot encoding for categorial (all) features
		categorial = data.columns.values.tolist()
		data = pd.get_dummies(data, columns = categorial)

		return data, labels

	# apply preprocessing to training and validation set
	train_data, train_labels = preprocess(breast_train)
	val_data, val_labels = preprocess(breast_val)

	# now save the .csv files
	train_data.to_csv(f'data/{path}breast_train_data.csv', index=False)
	train_labels.to_csv(f'data/{path}breast_train_labels.csv', index=False)

	val_data.to_csv(f'data/{path}breast_val_data.csv', index=False)
	val_labels.to_csv(f'data/{path}breast_val_labels.csv', index=False)


def preproc_census(path=""):
	'''Preprocesses census dataset '''
	census_train = pd.read_csv(f'data/{path}census_train.csv')
	census_val = pd.read_csv(f'data/{path}census_val.csv')

	def preprocess(data):
		# make binary labels for income column
		data['income'] = data['income'].str.replace('<=50K', '0')
		data['income'] = data['income'].str.replace('>50K', '1')
		data['income'] = data['income'].astype(int)

		# make array with labels, remove labels from dataframe
		labels = data['income'].copy()
		data = data.drop(['income'], axis=1)

		# use Min-max scaling for continuous features
		continuous = ['age','capital_gain','capital_loss','hr_per_week']
		data[continuous] = MinMaxScaler().fit_transform(data[continuous])

		# use One-hot encoding for categorial features
		categorial = [feat for feat in data.columns.values if feat not in continuous]
		data = pd.get_dummies(data, columns = categorial)

		return data, labels

	# apply preprocessing to training and validation set
	train_data, train_labels = preprocess(census_train)
	val_data, val_labels = preprocess(census_val)

	# now save the .csv files
	train_data.to_csv(f'data/{path}census_train_data.csv', index=False)
	train_labels.to_csv(f'data/{path}census_train_labels.csv', index=False)

	val_data.to_csv(f'data/{path}census_val_data.csv', index=False)
	val_labels.to_csv(f'data/{path}census_val_labels.csv', index=False)

def preproc_student(path=""):
	'''Preprocesses student dataset '''
	student_train = pd.read_csv(f'data/{path}student_train.csv')
	student_val = pd.read_csv(f'data/{path}student_val.csv')

	def preprocess(data):
		# convert student grade to pass or fail
		data.loc[data['G3'] < 10, 'G3'] = 0
		data.loc[data['G3'] > 9, 'G3'] = 1
		data['G3'] = data['G3'].astype(int)

		# make array with labels, remove labels from dataframe
		labels = data['G3'].copy()
		data = data.drop(['G3'], axis=1)

		# use Min-max scaling for continuous features
		continuous = ['age','absences','G1','G2']
		data[continuous] = MinMaxScaler().fit_transform(data[continuous])

		# use One-hot encoding for categorial features
		categorial = [feat for feat in data.columns.values if feat not in continuous]
		data = pd.get_dummies(data, columns = categorial)

		return data, labels

	# split into training and validation set, same ratio as Adult set
	train_data, train_labels = preprocess(student_train)
	val_data, val_labels = preprocess(student_val)

	# now save the .csv files
	train_data.to_csv(f'data/{path}student_train_data.csv', index=False)
	train_labels.to_csv(f'data/{path}student_train_labels.csv', index=False)

	val_data.to_csv(f'data/{path}student_val_data.csv', index=False)
	val_labels.to_csv(f'data/{path}student_val_labels.csv', index=False)


def preproc_heart(path=""):
	'''Preprocess heart disease dataset'''
	heart_train = pd.read_csv(f'data/{path}heart_train.csv')
	heart_val = pd.read_csv(f'data/{path}heart_val.csv')

	def preprocess(data):
		# convert diagnosis to presence or absence
		data.loc[data['num'] > 0, 'num'] = 1
		data.loc[data['num'] == 0, 'num'] = 0

		to_type_int = [col for col in data.columns.values.tolist() if col != 'oldpeak']
		data[to_type_int] = data[to_type_int].astype(int)

		# make array with labels, remove labels from dataframe
		labels = data['num'].copy()
		data = data.drop(['num'], axis=1)

		# use Min-max scaling for continuous features
		continuous = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
		data[continuous] = MinMaxScaler().fit_transform(data[continuous])

		# use One-hot encoding for categorial features
		categorial = [feat for feat in data.columns.values if feat not in continuous]
		data = pd.get_dummies(data, columns = categorial)

		return data, labels

	# apply preprocessing to training and validation set
	train_data, train_labels = preprocess(heart_train)
	val_data, val_labels = preprocess(heart_val)

	# now save the .csv files
	train_data.to_csv(f'data/{path}heart_train_data.csv', index=False)
	train_labels.to_csv(f'data/{path}heart_train_labels.csv', index=False)

	val_data.to_csv(f'data/{path}heart_val_data.csv', index=False)
	val_labels.to_csv(f'data/{path}heart_val_labels.csv', index=False)


if __name__ == "__main__":
	preproc_breast()
	preproc_student()
	preproc_census()
	preproc_heart()
	exit()
	preproc_breastcanc(path="syn/")
	preproc_student(path="syn/")
	#preproc_census(path="syn/")
