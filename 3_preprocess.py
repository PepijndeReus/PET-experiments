"""
This python script will pepare the (synthetic) data for the machine learning
task.

Please note that this data has already been cleaned (see 1_cleaning.py). Data
should be present in the 'data/' folder.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preproc_breastcanc(path=""):
	'''Preprocesses breast-cancer dataset '''
	breast = pd.read_csv(f'data/{path}breast_train.csv')

	# make binary labels for recurrence
	breast['class'] = breast['class'].str.replace('no-recurrence-events', '0')
	breast['class'] = breast['class'].str.replace('recurrence-events', '1')
	breast['class'] = breast['class'].astype(int)

	# split training data and label
	breast_data = breast.loc[:,breast.columns != 'class']
	breast_target = breast['class']

	# use One-hot encoding for categorial (all) features
	columns = breast_data.columns.values.tolist()
	breast_data = pd.get_dummies(breast_data, columns = columns)

	# split into training and validation set, same ratio as Adult set
	train_data = breast_data[:int(len(breast_data) * (2/3))]
	val_data = breast_data[int(len(breast_data) * (2/3)):]
	train_label = breast_target[:int(len(breast_data) * (2/3))]
	val_label = breast_target[int(len(breast_data) * (2/3)):]

	# now save the .csv files
	train_data.to_csv(f'data/{path}breast_train_data.csv', index=False)
	train_label.to_csv(f'data/{path}breast_train_labels.csv', index=False)

	# and for the validation set
	val_data.to_csv(f'data/{path}breast_val_data.csv', index=False)
	val_label.to_csv(f'data/{path}breast_val_labels.csv', index=False)


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
	census_train, labels_train = preprocess(census_train)
	census_val, labels_val = preprocess(census_val)

	# now save the .csv files
	census_train.to_csv(f'data/{path}census_train_data.csv', index=False)
	labels_train.to_csv(f'data/{path}census_train_labels.csv', index=False)

	census_val.to_csv(f'data/{path}census_val_data.csv', index=False)
	labels_val.to_csv(f'data/{path}census_val_labels.csv', index=False)

def preproc_student(path=""):
	'''Preprocesses student dataset '''
	student = pd.read_csv(f'data/{path}student_train.csv')

	# convert student grade to pass or fail
	student.loc[student['G3'] < 10, 'G3'] = 0
	student.loc[student['G3'] > 9, 'G3'] = 1
	student['G3'] = student['G3'].astype(int)

	# use Min-max scaling for continuous features
	continuous = ['age','absences','G1','G2']
	student[continuous] = MinMaxScaler().fit_transform(student[continuous])

	# split training data and label
	student_data = student.loc[:,student.columns != 'G3']
	student_target = student['G3']

	# use One-hot encoding for categorial features
	categorial = [feat for feat in student_data.columns.values if feat not in continuous]
	student_data = pd.get_dummies(student_data, columns = categorial)

	# split into training and validation set, same ratio as Adult set
	student_train = student_data[:int(len(student_data) * (2/3))]
	student_val = student_data[int(len(student_data) * (2/3)):]
	grade_train = student_target[:int(len(student_data) * (2/3))]
	grade_val = student_target[int(len(student_data) * (2/3)):]

	# now save the .csv files
	student_train.to_csv(f'data/{path}student_train_data.csv', index=False)
	grade_train.to_csv(f'data/{path}student_train_labels.csv', index=False)

	# and for the validation set
	student_val.to_csv(f'data/{path}student_val_data.csv', index=False)
	grade_val.to_csv(f'data/{path}student_val_labels.csv', index=False)


def preproc_heart(path=""):
	'''Preprocess heart disease dataset'''
	heart = pd.read_csv(f'data/{path}heart_train.csv')

	# convert diagnosis to presence or absence
	heart.loc[heart['num'] > 0, 'num'] = 1
	heart.loc[heart['num'] == 0, 'num'] = 0
	to_type_int = heart.columns.values.tolist().remove('exang')
	heart[to_type_int] = heart[to_type_int].astype(int)
	print(heart)
	categorical = ['sex', 'cp']
	heart['num'] = heart['num'].astype(int)

	# Todo afmaken

if __name__ == "__main__":
	preproc_breastcanc()
	preproc_student()
	#preproc_census()
	exit()
	preproc_breastcanc(path="syn/")
	preproc_student(path="syn/")
	#preproc_census(path="syn/")
