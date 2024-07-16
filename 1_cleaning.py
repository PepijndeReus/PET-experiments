"""
This script will remove redundant and unnecessary rows and columns from the data.
"""
import os
import pandas as pd


def clean_breastcanc():
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

	# Data is ordered, shuffle data
	new = breast.sample(frac=1, random_state=1, axis=0)

	os.makedirs('data', exist_ok=True)
	new.to_csv('data/breast_train.csv', index=False)


def clean_census():
	'''Clean adult dataset, and save it to a .csv'''
	# load data, use columns from census.names file
	columns = ["age", "type_employer", "fnlwgt", "education", "education_num", "marital",
			   "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
			   "hr_per_week", "country", "income"]
	census = pd.read_csv("raw/adult.data", sep=",\\s", names=columns, na_values=["?"],
						 engine='python')

	# delete unnecessary (redundant and highly unique) columns
	census = census.drop('education_num', axis=1)
	census = census.drop('fnlwgt', axis=1)

	# since only 1 entry for entire set, remove this row
	census = census[census.country != 'Holand-Netherlands']

	# drop NA values from data set
	census = census.dropna()

	os.makedirs('data', exist_ok=True)
	census.to_csv('data/census_train.csv', index=False)

	#################### and now for the validation set ####################
	# load data, use columns from census.names file
	columns = ["age", "type_employer", "fnlwgt", "education", "education_num", "marital",
			   "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
			   "hr_per_week", "country", "income"]
	census = pd.read_csv("raw/adult.test", sep=",\\s", names=columns, na_values=["?"],
						 engine='python')

	# delete unnecessary columns
	census = census.drop('education_num', axis=1)
	census = census.drop('fnlwgt', axis=1)

	# drop NA values from data set
	census = census.dropna()

	# replace different values from training set
	census['income'] = census['income'].str.replace('<=50K.', '<=50K', regex=True)
	census['income'] = census['income'].str.replace('>50K.', '>50K', regex=True)

	census.to_csv('data/census_val.csv', index=False)


def clean_student():
	'''Clean student dataset, and save it to a .csv'''
	# change separator for Student data set
	data = pd.read_csv('raw/student-por.csv', sep=';')
	os.makedirs('data', exist_ok=True)
	data.to_csv('data/student_train.csv', index=False)


def clean_heart():
	'''Clean heart disease dataset, and save it to a .csv'''
	# load data, use columns from heart-disease.names file
	columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
			   "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
	heart = pd.read_csv('raw/processed.cleveland.data', names=columns, na_values=["?"])
	heart = heart.dropna()
	os.makedirs('data', exist_ok=True)
	heart.to_csv('data/heart_train.csv', index=False)


if __name__ ==  "__main__":
	clean_heart()
	clean_breastcanc()
	clean_census()
	clean_student()
