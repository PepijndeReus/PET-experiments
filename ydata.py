#Uncomment to install ydata-synthetic lib
#!pip install ydata-synthetic

from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import pandas as pd
import numpy as np

#load data
data = pd.read_csv('Data/Student/student-por.csv')

# Extract numerical columns
num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Extract categorical columns
cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

# Defining the training parameters
batch_size = 500
epochs = 50
learning_rate = 2e-4
beta_1 = 0.5
beta_2 = 0.9

ctgan_args = ModelParameters(batch_size=batch_size,
                             lr=learning_rate,
                             betas=(beta_1, beta_2))

train_args = TrainParameters(epochs=epochs)

#create and train CTGAN
synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

#generate new synthetic data
synth_data = synth.sample(649)
print(synth_data)

synth_data.to_csv("Data/Student/synthetic_data/Student_synthetic_ydata_ctgan.csv")