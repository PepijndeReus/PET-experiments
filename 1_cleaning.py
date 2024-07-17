"""
This script will clean the data:
  * remove redundant rows
  * remove highly unique columns
  * drop NA values
  * shuffle ordered data
  * make labels training and valudation equal
  * if not already done, split into training and validation data
"""
import os
import pandas as pd


def clean_breast():
	'''Clean breast-cancer dataset, and save it to a .csv'''
	columns = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig",
			   "breast", "breast-quad", "irradiat"]
	breast = pd.read_csv("raw/breast-cancer.data", sep=",", names=columns, na_values=["?"],
 						 engine='python')

	# drop NA values from data set
	breast = breast.dropna()

	# since only 1 entry for entire set, remove this row
	breast = breast[breast.age != '20-29']
	breast = breast[breast['inv-nodes'] != '24-26']

	# data is ordered, shuffle data
	breast_shuffled = breast.sample(frac=1, random_state=1, axis=0)

	# split into training and validation set, same ratio as Adult set
	breast_train = breast_shuffled[:int(len(breast_shuffled) * (2/3))]
	breast_val = breast_shuffled[int(len(breast_shuffled) * (2/3)):]

	# save training and validation set
	os.makedirs('data', exist_ok=True)
	breast_train.to_csv('data/breast_train.csv', index=False)
	breast_val.to_csv('data/breast_val.csv', index=False)


def clean_census():
	'''Clean adult dataset, and save it to a .csv'''
	# load data, use columns from census.names file
	columns = ["age", "type_employer", "fnlwgt", "education", "education_num", "marital",
			 "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
			 "hr_per_week", "country", "income"]
	census_train = pd.read_csv("raw/adult.data", sep=",\\s", names=columns, na_values=["?"],
							   engine='python')
	census_val = pd.read_csv("raw/adult.test", sep=",\\s", names=columns, na_values=["?"],
							 engine='python')
	# replace different values from training set
	census_val['income'] = census_val['income'].str.replace('<=50K.', '<=50K', regex=True)
	census_val['income'] = census_val['income'].str.replace('>50K.', '>50K', regex=True)

	# drop NA values from data set
	census_train = census_train.dropna()
	census_val = census_val.dropna()

	# delete unnecessary (redundant and highly unique) columns
	census_train = census_train.drop('education_num', axis=1)
	census_val = census_val.drop('education_num', axis=1)
	census_train = census_train.drop('fnlwgt', axis=1)
	census_val = census_val.drop('fnlwgt', axis=1)

	# since only 1 entry for entire set, remove this row
	census_train = census_train[census_train.country != 'Holand-Netherlands']
	census_val = census_val[census_val.country != 'Holand-Netherlands']

	# save training and validation set
	os.makedirs('data', exist_ok=True)
	census_train.to_csv('data/census_train.csv', index=False)
	census_val.to_csv('data/census_val.csv', index=False)


def clean_student():
	'''Clean student dataset, and save it to a .csv'''
	# change separator for Student data set
	student = pd.read_csv('raw/student-por.csv', sep=';')

	# split into training and validation set, same ratio as Adult set
	student_train = student[:int(len(student) * (2/3))]
	student_val = student[int(len(student) * (2/3)):]

	# save training and validation set
	os.makedirs('data', exist_ok=True)
	student_train.to_csv('data/student_train.csv', index=False)
	student_val.to_csv('data/student_val.csv', index=False)


def clean_heart():
	'''Clean heart disease dataset, and save it to a .csv'''
	# load data, use columns from heart-disease.names file
	columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
			 "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
	heart = pd.read_csv('raw/processed.cleveland.data', names=columns, na_values=["?"])

	# drop NA values from data set
	heart = heart.dropna()

	# split into training and validation set, same ratio as Adult set
	heart_train = heart[:int(len(heart) * (2/3))]
	heart_val = heart[int(len(heart) * (2/3)):]

	# save training and validation set
	os.makedirs('data', exist_ok=True)
	heart_train.to_csv('data/heart_train.csv', index=False)
	heart_val.to_csv('data/heart_val.csv', index=False)

if __name__ =="__main__":
	clean_heart()
	clean_breast()
	clean_census()
	clean_student()
