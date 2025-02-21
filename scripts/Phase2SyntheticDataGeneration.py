"""
This file contains the code that is used in main.py to generate synthetic data.
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
    data_loc = f'data/{label}_train.csv'

    # settings for DataSynthesizer
    threshold = 42 # Threshold for categorical
    epsilon = 0.05 # Differential privacy
    degree_of_bayesian_network = 2 # amount of parent nodes for Bayesian network
    num_tuples_to_generate = int(len(data))

    # location for output files
    description_file = f'./description_{label}.json'
    synthetic_data_loc = f'data/syn/{label}_train_datasynthesizer.csv'

    # describe data set
    describer = DataDescriber(category_threshold=threshold)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=data_loc,
                                                            epsilon=epsilon,
                                                            k=degree_of_bayesian_network)

    describer.save_dataset_description_to_file(description_file)

    # Generate data set
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)

    # Save data set
    generator.save_synthetic_data(synthetic_data_loc)


def dpctgan(label, dict):
    '''Create synthetic data and save in 'data/syn/' '''
    data = pd.read_csv(f'data/{label}_train.csv')

    discr_cols = dict['discr_cols']

    changed_cols = []
    for col in discr_cols:
        if data[col].dtype == 'int64':
            data[col] = data[col].apply(str)
            changed_cols.append(col)
            # print(data[col].dtype, col)

    # Fit model
    dpctgan = DPCTGAN(verbose=True, batch_size=dict['batch_size'], 
                      epochs=dict['epochs'], generator_lr=dict['learning_rate'],
                      discriminator_lr=dict['learning_rate'], cuda=False, target_epsilon=0.05) #steps, target_epsilon/delta
    dpctgan.fit(data, discr_cols)

    for col in changed_cols:
        data[col] = data[col].apply(int)

    # Save data
    os.makedirs('data/syn', exist_ok=True)
    synthetic_data = dpctgan.sample(len(data))
    synthetic_data.to_csv(f'data/syn/{label}_train_dpctgan.csv', index=False)


def ydata(label, dict):
    # Load the data
    data = pd.read_csv(f'data/{label}_train.csv')

    # Define the training parameters
    ctgan_args = ModelParameters(batch_size=dict['batch_size'], lr=dict['learning_rate'], betas=(dict['beta_1'], dict['beta_2']))
    train_args = TrainParameters(epochs=dict['epochs'])

    # for census, one iteration takes >10 minutes so we limit the epochs to 5.
    if label == 'census':
        train_args = TrainParameters(epochs=5)

    # Initialize and train the synthesizer
    synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
    num_cols = [col for col in data.columns if col not in dict['discr_cols']]
    synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=dict['discr_cols'])

    # Save data
    os.makedirs('data/syn', exist_ok=True)
    synthetic_data = synth.sample(len(data))
    synthetic_data.to_csv(f'data/syn/{label}_train_ydata.csv', index=False)


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
    syn_model = Plugins().get("adsgan", n_iter=300)

    # for census, one iteration takes 520 seconds so we limit the iterations.
    if label == 'census':
        syn_model = Plugins().get("adsgan", n_iter=3)
    
    syn_model.fit(loader)

    # Save data
    os.makedirs('data/syn', exist_ok=True)
    synthetic_data = syn_model.generate(len(data)).dataframe()
    synthetic_data.to_csv(f'data/syn/{label}_train_synthcity.csv', index=False)


def ctgan(label, dict):
    """Generate synthetic data with SDV CTGAN"""
    # Load the data
    data = pd.read_csv(f'data/{label}_train.csv')

    # Initialize and train the CTGAN model
    ctgan = CTGAN(epochs=dict['epochs'])
    ctgan.fit(data, dict['discr_cols'])

    # Save data
    os.makedirs('data/syn', exist_ok=True)
    synthetic_data = ctgan.sample(len(data))
    synthetic_data.to_csv(f'data/syn/{label}_train_ctgan.csv', index=False)


def nbsynthetic(label, dict):
    """
    Generate synthetic data using nbsynthetic
    """
    # Load the data
    data_without_strings = pd.read_csv(f'data/{label}_train.csv')

    # change strings
    data = data_without_strings

    # Initialize and use SmartBrain for encoding
    SB = SmartBrain()

    # Generate synthetic data
    newdf = synthetic_data(
        GAN,
        data,
        samples=len(data)
    )

    # Save data
    os.makedirs('data/syn', exist_ok=True)
    newdf.to_csv(f'data/syn/{label}_train_nbsynthetic.csv', index=False)


if __name__ == "__main__":
    input_data = str(input("Which dataset would you like to synthesise?\nPick: breast, census, heart or student.")).lower()
    input_method = str(input("Which synthetic data generator would you like to synthesise?\nPick: dpctgan, ydata, ctgan, synthcity, nbsynthetic or datasynthesizer.")).lower()

    # dicts are similar to main.py
    # this function is created so that Phase2 can be run as a separate Python file
    dict = {
        'breast': {
            'discr_cols': ['age', 'class', 'menopause', 'tumor-size',
                            'inv-nodes', 'node-caps', 'deg-malig',
                            'breast', 'breast-quad', 'irradiat'],
            'target_column': 'class',
            'other_var': 23,
            'batch_size': 10,
            'epochs': 300,
            'learning_rate': 2e-4,
            'beta_1': 0.5,
            'beta_2': 0.9
        },
        'student': {
            'discr_cols': ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu','Fedu',
                            'Mjob','Fjob','reason','guardian','traveltime', 'studytime',
                            'failures','schoolsup','famsup','paid','activities','nursery','higher',
                            'internet','romantic','famrel','freetime','goout','Dalc','Walc',
                            'health','absences','G1','G2','G3'],
            'target_column': 'G3',
            'other_var': 23,
            'batch_size': 10,
            'epochs': 300,
            'learning_rate': 2e-4,
            'beta_1': 0.5,
            'beta_2': 0.9
        },
        'census': {
            'discr_cols': ['age','type_employer','education','marital','occupation','relationship',
                            'race','sex','capital_gain','capital_loss','hr_per_week','country',
                            'income'],
            'target_column': 'income',
            'other_var': 23,
            'batch_size': 10,
            'epochs': 300,
            'learning_rate': 2e-4,
            'beta_1': 0.5,
            'beta_2': 0.9
        },
        'heart': {
            'discr_cols': ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
                            'oldpeak','slope','ca','thal','num'],
            'target_column': 'num',
            'other_var': 23,
            'batch_size': 10,
            'epochs': 300,
            'learning_rate': 2e-4,
            'beta_1': 0.5,
            'beta_2': 0.9
        }
    }

    # pick right data
    if input_data == 'breast':
        hyperparam = dict['breast']
    elif input_data == 'census':
        hyperparam = dict['census']
    elif input_data == 'heart':
        hyperparam = dict['heart']
    elif input_data == 'student':
        hyperparam = dict['student']
    else:
        print("The dataset you entered is invalid. Please try again")
        raise KeyError('data not in list of datasets')
    
    # pick right method
    if input_method == 'dpctgan':
        dpctgan(input_data, hyperparam)
    elif input_method == 'ydata':
        ydata(input_data, hyperparam)
    elif input_method == 'ctgan':
        ctgan(input_data, hyperparam)
    elif input_method == 'synthcity':
        synthcity(input_data, hyperparam)
    elif input_method == 'nbsynthetic':
        nbsynthetic(input_data, hyperparam)
    elif input_method == 'datasynthesizer':
        datasynthesizer(input_data, hyperparam)
    else:
        print("The method you entered is invalid. Please try again")
        raise KeyError('method not in list of synthetic data generators')

