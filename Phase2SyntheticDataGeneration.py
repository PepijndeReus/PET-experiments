"""
This file creates the synthetic data using DP-CTGAN
"""
import os
import pandas as pd
import numpy as np

# stdlib
import sys
import warnings

#ydata
# Import the necessary modules
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# synthcity
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

# sdv_ctgan
from ctgan import CTGAN

# dp-ctgan
from dp_ctgan.dpctgan import DPCTGAN

# nbsynthetic
from nbsynthetic.data import input_data
from nbsynthetic.data_preparation import SmartBrain
#create a GAN instance
from nbsynthetic.vgan import GAN
#generate synthetic dataset
from nbsynthetic.synthetic import synthetic_data

# DataSynthesizer
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
# unused: from DataSynthesizer.ModelInspector import ModelInspector

def datasynthesizer(label, dict):
	"""
	Create synthetic data with DataSynthesizer.
	Save data in 'data/syn/'
	"""

	# load data, set mode
	data = pd.read_csv(f'data/{label}_train.csv')

	# settings for DataSynthesizer
	mode = 'correlated_attribute_mode'
	threshold = 42 # Threshold for categorical
	epsilon = 0 # Differential privacy
	degree_of_bayesian_network = 2 # amount of parent nodes for Bayesian network
	num_tuples_to_generate = len(data)

	# Specify categorical attributes
    # categorical_attributes = {'type_employer': True, 'education': True}

	# location for output files
	description_file = f'./output/description_{label}.json'
	synthetic_data_loc = f'data/syn/{label}_train_datasynthesizer.csv'

	# describe data set
	describer = DataDescriber(category_threshold=threshold)
	describer.describe_dataset_in_correlated_attribute_mode(dataset_file=data, 
															epsilon=epsilon, 
															k=degree_of_bayesian_network)
	describer.save_dataset_description_to_file(description_file)

	# Generate data set
	generator = DataGenerator()
	generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)

	# Save data set
	generator.save_synthetic_data(synthetic_data_loc)

	# Compare unique values
	synthetic_data = pd.read_csv(synthetic_data_loc)
	for col in data.columns:
		real_unique = set(data[col].unique())
		synthetic_unique = set(synthetic_data[col].unique())
		if real_unique != synthetic_unique:
			print(f"{col} differs")
			print(f"Real: {real_unique}")
			print(f"Synthetic: {synthetic_unique}")


def dpctgan(label, dict):
	'''Create synthetic data and save in 'data/syn/' '''
	data = pd.read_csv(f'data/{label}_train.csv')

	discr_cols = dict['discr_cols']

	changed_cols = []
	for col in discr_cols:
		if data[col].dtype == 'int64':
			data[col] = data[col].apply(str)
			changed_cols.append(col)
			print(data[col].dtype, col)

	# Fit model
	dpctgan = DPCTGAN(verbose=True, batch_size=dict['batch_size'], 
					  epochs=dict['epochs'], generator_lr=dict['learning_rate'],
					  discriminator_lr=dict['learning_rate'], cuda=False) #steps, target_epsilon/delta
	dpctgan.fit(data, discr_cols)

	for col in changed_cols:
		data[col] = data[col].apply(int)

	# Save data
	os.makedirs('data/syn', exist_ok=True)
	synthetic_data = dpctgan.sample(len(data))
	synthetic_data.to_csv(f'data/syn/{label}_train_dpctgan.csv', index=False)

	# Compare unique values
	for col in data.columns:
		real_unique = set(data[col].unique())
		synthetic_unique = set(synthetic_data[col].unique())
		if real_unique != synthetic_unique:
			print(f"{col} differs")
			print(f"Real: {real_unique}")
			print(f"Synthetic: {synthetic_unique}")

def ydata(label, dict):
	# Load the data
	data = pd.read_csv(f'data/{label}_train.csv')

	# Extract numerical columns
	#num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

	# Extract categorical columns
	#cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

	#print("num cols", num_cols, "\n")
	#print("cat_cols", cat_cols, "\n")
	#print(dict['discr_cols'])

	# Define the training parameters
	ctgan_args = ModelParameters(batch_size=dict['batch_size'], lr=dict['learning_rate'], betas=(dict['beta_1'], dict['beta_2']))
	train_args = TrainParameters(epochs=dict['epochs'])

	# Initialize and train the synthesizer
	synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
	num_cols = [col for col in data.columns if col not in dict['discr_cols']]
	synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=dict['discr_cols'])

	# Save data
	os.makedirs('data/syn', exist_ok=True)
	synthetic_data = synth.sample(len(data))
	synthetic_data.to_csv(f'data/syn/{label}_train_ydata.csv', index=False)

	# Compare unique values
	for col in data.columns:
		real_unique = set(data[col].unique())
		synthetic_unique = set(synthetic_data[col].unique())
		if real_unique != synthetic_unique:
			print(f"{col} differs")
			print(f"Real: {real_unique}")
			print(f"Synthetic: {synthetic_unique}")

def synthcity(label, dict):
	"""
	Generate synthetic data using Synthcity.
	"""
	# Load the data
	data = pd.read_csv(f'data/{label}_train.csv')

	# Prepare data loader
	loader = GenericDataLoader(
		data,
		target_column=dict['target_column'])

	# Initialize and train the synthetic model
	syn_model = Plugins().get("adsgan")
	syn_model.fit(loader)

	# Save data
	os.makedirs('data/syn', exist_ok=True)
	synthetic_data = syn_model.generate(len(data)).dataframe()
	synthetic_data.to_csv(f'data/syn/{label}_train_synthcity.csv', index=False)

	# Compare unique values
	for col in data.columns:
		real_unique = set(data[col].unique())
		synthetic_unique = set(synthetic_data[col].unique())
		if real_unique != synthetic_unique:
			print(f"{col} differs")
			print(f"Real: {real_unique}")
			print(f"Synthetic: {synthetic_unique}")

def ctgan(label, dict):
	"""Generate synthetic data with SDV CTGAN"""
	# Load the data
	data = pd.read_csv(f'data/{label}_train.csv')

	# Identify discrete columns (assuming all non-numeric columns are discrete)
	#discrete_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

	# Initialize and train the CTGAN model
	ctgan = CTGAN(epochs=dict['epochs'])
	ctgan.fit(data, dict['discr_cols'])

	# Save data
	os.makedirs('data/syn', exist_ok=True)
	synthetic_data = ctgan.sample(len(data))
	synthetic_data.to_csv(f'data/syn/{label}_train_ctgan.csv', index=False)

	# Compare unique values
	for col in data.columns:
		real_unique = set(data[col].unique())
		synthetic_unique = set(synthetic_data[col].unique())
		if real_unique != synthetic_unique:
			print(f"{col} differs")
			print(f"Real: {real_unique}")
			print(f"Synthetic: {synthetic_unique}")

def nbsynthetic(label, dict):
	"""
	Generate synthetic data using nbsynthetic
	"""
	# Load the data
	data = pd.read_csv(f'data/{label}_train.csv')

	# Load the data
	#df = input_data(file_name, decimal=decimal)

	# Initialize and use SmartBrain for encoding
	SB = SmartBrain()
	#df_encoded = SB.nbEncode(data)

	# Generate synthetic data
	newdf = synthetic_data(
		GAN,
		#df_encoded,
		data,
		samples=len(data)
	)
	#synt_data = SB.nbDecode(newdf)
	print(newdf)

	# Save data
	os.makedirs('data/syn', exist_ok=True)
	newdf.to_csv(f'data/syn/{label}_train_nbsynthetic.csv', index=False)


if __name__ == "__main__":
	brcanc = {
				'discr_cols': ['age', 'class', 'menopause', 'tumor-size',
		 						  'inv-nodes', 'node-caps', 'deg-malig',
								  'breast', 'breast-quad', 'irradiat'],
				'other_var': 23,
				'batch_size': 10,
				'epochs': 10,
				'learning_rate': 2e-4,
				'beta_1': 0.5,
				'beta_2': 0.9
			}

	student = {
				'discr_cols': ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu','Fedu', 'Mjob','Fjob','reason','guardian','traveltime', 'studytime', 'failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3'],
				'other_var': 23,
				'batch_size': 10,
				'epochs': 10,
				'learning_rate': 2e-4,
				'beta_1': 0.5,
				'beta_2': 0.9
			}

	census = {
				'discr_cols': ['age','type_employer','education','marital','occupation','relationship','race','sex','capital_gain','capital_loss','hr_per_week','country','income'],
				'other_var': 23,
				'batch_size': 10,
				'epochs': 10,
				'learning_rate': 2e-4,
				'beta_1': 0.5,
				'beta_2': 0.9
			}

	heart = {
				'discr_cols': ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'],
				'other_var': 23,
				'batch_size': 10,
				'epochs': 10,
				'learning_rate': 2e-4,
				'beta_1': 0.5,
				'beta_2': 0.9
			}
	#dpctgan('breast', brcanc)
	#ydata('breast', brcanc)
	#sdv_ctgan('breast', brcanc)

	#synthcity('breast', brcanc) # problemen met goggle
	#dpctgan('census', census)
	ydata('heart', heart)
	sdv_ctgan_ctgan('heart', heart)
	synthcity('heart', heart)
	exit()




	dpctgan(
		'breast',
		['age', 'class', 'menopause', 'tumor-size',
		 'inv-nodes', 'node-caps', 'deg-malig',
		 'breast', 'breast-quad', 'irradiat'])
	dpctgan(
		'student',
		['school', 'sex', 'address', 'famsize',
		 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
		 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
		 'paid', 'activities', 'nursery',
		 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
		 'Walc', 'health', 'G1', 'G2', 'G3'])

	#create_synthetic(
	#	'census',
	#	['type_employer', 'education',
	#	 'marital', 'occupation', 'relationship',
	#	 'race', 'sex', 'country', 'income'])
