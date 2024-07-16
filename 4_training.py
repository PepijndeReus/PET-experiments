"""
Train the model on the data
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import torch
import pandas as pd
import numpy as np
import crypten # MPC
import crypten.optim.sgd as soptim
import crypten.communicator as comm
#import crypten.optimizer as coptim
from torch import nn
from torch import optim
from sklearn import metrics
from model import ExampleNet

def batch(it1, it2, size=1):
	'''
	Batch generator
	https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
	'''
	length = len(it1)
	for ndx in range(0, length, size):
		yield it1[ndx:min(ndx+size, length)], it2[ndx:min(ndx+size, length)]

def train(data_set, epochs=100, learning_rate=.001, batch_size=15):
	'''Train the plaintext model'''
	### Load data ###
	# Training
	train_data = torch.tensor(pd.read_csv(f'data/{data_set}_train_data.csv').values).float()
	train_labels = torch.tensor(pd.read_csv(f'data/{data_set}_train_labels.csv').values).float()

	# Valdating
	val_data = torch.tensor(pd.read_csv(f'data/{data_set}_val_data.csv').values).float()
	val_labels = torch.tensor(pd.read_csv(f'data/{data_set}_val_labels.csv').values).float()

	### Define model ###
	model_plaintext = ExampleNet(n_inputs=len(train_data[0]))
	model_plaintext.reset()
	loss_func = nn.MSELoss()
	optimizer = optim.SGD(model_plaintext.parameters(), lr=learning_rate)

	for i in range(1, epochs+1):
		### Train the model ###
		model_plaintext.train()

		for input_train, target_train in batch(train_data, train_labels, size=batch_size):
			optimizer.zero_grad()

			# Forward pass
			output_train = model_plaintext(input_train)
			loss_value = loss_func(output_train, target_train)

			# Backward pass
			loss_value.backward()
			optimizer.step()

		### Compute accuracy ###
		if i==epochs:
			model_plaintext.eval()
			cm = [[0,0],[0,0]]
			# Validating, so dont need to calculate gradients
			with torch.no_grad():
				for input_val, target_val in batch(val_data, val_labels, size=1):
					pred = int(round(model_plaintext(input_val).item(),0))
					targ = int(target_val.item())
					cm[1-targ][1-pred] += 1

			# Save results to csv
			a = {
				'tp': [cm[0][0]],
				'fn': [cm[0][1]],
				'fp': [cm[1][0]],
				'tn': [cm[1][1]]
			}

			datafile = pd.DataFrame(a)
			datafile.to_csv(f'measurements/performance_{data_set.replace("/", "_")}.csv',
							index=False, mode='a')


def split(data_set, seed=0):
	'''Split data for MPC scenario'''
	train_data = torch.tensor(pd.read_csv(f'data/{data_set}_train_data.csv').values).float()
	train_labels = torch.tensor(pd.read_csv(f'data/{data_set}_train_labels.csv').values).float()

	np.random.seed(seed)
	mask = np.random.rand(len(train_labels)) <= .5
	train_alice_data = train_data[mask]
	train_bob_data = train_data[~mask]

	# We need to put the labels in the same order
	train_alice_labels = train_labels[mask]
	train_bob_labels = train_labels[~mask]
	train_labels = torch.cat([train_alice_labels, train_bob_labels])

	os.makedirs('data/mpc/', exist_ok=True)
	torch.save(train_alice_data, f"data/mpc/{data_set}_train_alice_data.pt")
	torch.save(train_bob_data, f"data/mpc/{data_set}_train_bob_data.pt")
	torch.save(train_labels, f"data/mpc/{data_set}_train_labels.pt")


@crypten.mpc.run_multiprocess(world_size=2)
def train_enc(data_set, epochs=100, learning_rate=.001, batch_size=15):
	crypten.common.serial.register_safe_class(ExampleNet)
	### Load data ###
	# Training
	train_data = crypten.cat([
		crypten.load_from_party(f'data/mpc/{data_set}_train_alice_data.pt', src=0),
		crypten.load_from_party(f'data/mpc/{data_set}_train_bob_data.pt', src=1)
	]).squeeze()
	# Labels are public
	train_labels = torch.load(f'data/mpc/{data_set}_train_labels.pt').float()

	# Valdating
	val_data = torch.tensor(pd.read_csv(f'data/{data_set}_val_data.csv').values).float()
	val_labels = torch.tensor(pd.read_csv(f'data/{data_set}_val_labels.csv').values).float()

	### Define model ###
	model_plaintext = ExampleNet(n_inputs=len(train_data[0]))
	model_plaintext.reset()
	model_enc = crypten.nn.from_pytorch(model_plaintext, torch.empty((1, len(train_data[0]))))
	model_enc.encrypt()
	closs_func = crypten.nn.MSELoss()
	optimizer = soptim.SGD(model_enc.parameters(), lr=learning_rate)

	for i in range(1, epochs+1):
		### Train the model ###
		model_enc.train()

		for input_train, target_train in batch(train_data, train_labels, size=batch_size):
			optimizer.zero_grad()

			# Forward pass
			output_enc = model_enc(input_train)
			target_enc = crypten.cryptensor(target_train, requires_grad=True)
			loss_value = closs_func(output_enc, target_enc)

			# Backward pass
			loss_value.backward()
			optimizer.step()

		### Compute accuracy ###
		if i==epochs:
			model_enc.eval()
			cm = [[0,0],[0,0]]
			# Validating, so dont need to calculate gradients
			with torch.no_grad():
				for input_val, target_val in batch(val_data, val_labels, size=1):
					input_enc = crypten.cryptensor(input_val)
					pred = int(round(model_enc(input_enc).get_plain_text().item(),0))
					targ = int(target_val.item())
					cm[1-targ][1-pred] += 1

			# Save results to csv
			if comm.get().get_rank() == 0:
				a = {
					'tp': [cm[0][0]],
					'fn': [cm[0][1]],
					'fp': [cm[1][0]],
					'tn': [cm[1][1]]
				}

				datafile = pd.DataFrame(a)
				datafile.to_csv(f'measurements/performance_mpc_{data_set.replace("/", "_")}.csv',
							index=False, mode='a')



if __name__ == "__main__":
	train('student', 300, .05, 50)
	train('breast', 300, .05, 50)

	train('syn/student', 300, .5, 10)
	train('syn/breast', 300, .1, 50)

	split('student', seed=1)
	train_enc('student', 1, .5, 50)
	split('breast', seed=1)
	train_enc('student', 100, .5, 50)
