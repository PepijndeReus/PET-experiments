#install nbsynthetic 
#pip install git+https://github.com/NextBrain-ai/nbsynthetic.git

#packages
from nbsynthetic.data import input_data
from nbsynthetic.data_preparation import SmartBrain

#load data
df = input_data('student-por', decimal=',')
SB = SmartBrain() 
df = SB.nbEncode(df) #GAN input data

#create a GAN instance
from nbsynthetic.vgan import GAN

#generate synthetic dataset
from nbsynthetic.synthetic import synthetic_data

samples= 649 #number of samples we want to generate
newdf = synthetic_data(
    GAN, 
    df, 
    samples = samples
    )