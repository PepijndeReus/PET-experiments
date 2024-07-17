"""
This file creates the synthetic data using DP-CTGAN
"""
import os
import pandas as pd
from dp_ctgan.dpctgan import DPCTGAN

def syn_dpctgan(label, discr_cols, batch_size=10):
	'''Create synthetic data and save in 'data/syn/' '''
	data = pd.read_csv(f'data/{label}_train.csv')

	changed_cols = []
	for col in discr_cols:
		if data[col].dtype == 'int64':
			data[col] = data[col].apply(str)
			changed_cols.append(col)
			print(data[col].dtype, col)

	# Fit model
	dpctgan = DPCTGAN(verbose=True, batch_size=batch_size)
	dpctgan.fit(data, discr_cols)

	for col in changed_cols:
		data[col] = data[col].apply(int)

	# Save data
	os.makedirs('data/syn', exist_ok=True)
	new = dpctgan.sample(len(data))
	new.to_csv(f'data/syn/{label}_train.csv', index=False)

def syn_ydata(label, batch_size=500, epochs=50, learning_rate=2e-4, beta_1=0.5, beta_2=0.9, sample_size=649):
    # Load the data
    data = pd.read_csv(f'data/{label}_train.csv')
    
    # Extract numerical columns
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Extract categorical columns
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Define the training parameters
    ctgan_args = ModelParameters(batch_size=batch_size, lr=learning_rate, betas=(beta_1, beta_2))
    train_args = TrainParameters(epochs=epochs)
    
    # Initialize and train the synthesizer
    synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
    synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

	# Save data
	os.makedirs('data/syn', exist_ok=True)
	new = synth.sample(len(data))
	new.to_csv(f'data/syn/{label}_train.csv', index=False)

def syn_synthcity(label, target_column):
    """
    Generate synthetic data using Synthcity.
    """
    # Load the data
    data = pd.read_csv(f'data/{label}_train.csv')
    
    # Define X and y
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Prepare data loader
    loader = GenericDataLoader(
        data,
        target_column=target_column,
    )
    
    # Initialize and train the synthetic model
    syn_model = Plugins().get("adsgan")
    syn_model.fit(loader)
    
    # Save data
    synthetic_data = syn_model.generate(len(data)).dataframe()
	synthetic_data.to_csv(f'data/syn/{label}_train.csv', index=False)

def syn_sdv_ctgan(label, target_column, epochs=50):
	"""Generate synthetic data with SDV CTGAN"""
	# Load the data
    data = pd.read_csv(f'data/{label}_train.csv')

	# Identify discrete columns (assuming all non-numeric columns are discrete)
    discrete_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    
    # Initialize and train the CTGAN model
    ctgan = CTGAN(epochs=epochs)
    ctgan.fit(preprocessed_data, discrete_columns)

	# Save data
	os.makedirs('data/syn', exist_ok=True)
	synthetic_data = ctgan.sample(len(data))
	synthetic_data.to_csv(f'data/syn/{label}_train.csv', index=False)

def syn_nbsynthetic(label, decimal=','):
    """
	Generate synthetic data using nbsynthetic 
    """
	# Load the data
    data = pd.read_csv(f'data/{label}_train.csv')
    
	# Load the data
    #df = input_data(file_name, decimal=decimal)
    
    # Initialize and use SmartBrain for encoding
    SB = SmartBrain()
    df_encoded = SB.nbEncode(data)
    
    # Generate synthetic data
	os.makedirs('data/syn', exist_ok=True)
    newdf = synthetic_data(
        GAN, 
        df_encoded, 
        samples=len(data)
    )
    newdf.to_csv(f'data/syn/{label}_train.csv', index=False)


pass

if __name__ == "__main__":
	syn_dpctgan(
		'breast',
		['age', 'class', 'menopause', 'tumor-size',
		 'inv-nodes', 'node-caps', 'deg-malig',
		 'breast', 'breast-quad', 'irradiat'])
	syn_dpctgan(
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
