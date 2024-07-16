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

def syn_other(label):
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
