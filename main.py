import argparse
import os
import pyRAPL
import yaml
import pandas as pd

import Phase1Cleaning
import Phase2SyntheticDataGeneration
import Phase3Preprocessing
import Phase4MachineLearning

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
	#	name		= "name",
		description	= "Measure energy and utility of PETs",
		epilog		= "text ad bottom om help")

	parser.add_argument('-d', '--dataset', nargs="+", dest='label',
		choices=['student', 'breast', 'heart', 'census'],
		default=['student', 'breast', 'heart', 'census'])
	parser.add_argument('-s', '--generator', nargs="+", dest='generators',
		choices=['benchmark', 'dpctgan', 'ydata', 'ctgan', 'synthcity', 'nbsynthetic'],
		default=['benchmark'])
	parser.add_argument('-p', '--phases', nargs="+", dest='phases', type=int,
		choices=[1,2,3,4],
		default=[1,2,3,4])
	parser.add_argument('-t', '--ml_task', nargs="+", dest='tasks',
		choices=['knn', 'lr', 'nn'],
		default=['knn', 'lr', 'nn'])
	parser.add_argument('-n', '--amount', type=int, dest='amount', default=1)
	parser.add_argument('--version', action='version', version='%(prog)s 0.1')
	parser.add_argument('-v', '--verbose', action='store_true')
	# TODO add custom yaml files

	args = parser.parse_args()

	if not os.path.isdir('raw'):
		print("Data is not present in folder 'raw/'. Downloading the data now!")
		os.system('./download_raw.sh')
	"""if not all([False for val in [1,2,3,4] if val not in args.phases]):
		print("Warning: not executing all phases. Might result in FileNotFoundError.")
		confirmation = input("Do you want to continue? (y/n) ").lower()
		while confirmation not in ("y", "n"):
		confirmation = input("Invalid input. Please enter 'y' or 'n': ").lower()
		if confirmation == 'n': exit()
	"""

	vprint = print if args.verbose else lambda *a, **k: None

	pyRAPL.setup()
	os.makedirs('measurements', exist_ok=True)

	#config = yaml.save_load(open('config/main.yml'))
	#print(config)
	dict = {
		'breast': {
			'discr_cols': ['age', 'class', 'menopause', 'tumor-size',
						   'inv-nodes', 'node-caps', 'deg-malig',
						   'breast', 'breast-quad', 'irradiat'],
			'target_column': 'class',
			'other_var': 23,
			'batch_size': 10,
			'epochs': 1,
			'learning_rate': 2e-4,
			'beta_1': 0.5,
			'beta_2': 0.9
		},
		'student': {
			'discr_cols': ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu','Fedu', 'Mjob','Fjob','reason','guardian','traveltime', 'studytime', 'failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3'],
			'other_var': 23,
			'batch_size': 10,
			'epochs': 1,
			'learning_rate': 2e-4,
			'beta_1': 0.5,
			'beta_2': 0.9
		},
		'census': {
			'discr_cols': ['age','type_employer','education','marital','occupation','relationship','race','sex','capital_gain','capital_loss','hr_per_week','country','income'],
			'other_var': 23,
			'batch_size': 10,
			'epochs': 1,
			'learning_rate': 2e-4,
			'beta_1': 0.5,
			'beta_2': 0.9
		},
		'heart': {
			'discr_cols': ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'],
			'other_var': 23,
			'batch_size': 10,
			'epochs': 1,
			'learning_rate': 2e-4,
			'beta_1': 0.5,
			'beta_2': 0.9
		}
	}


	#exit()
	print(args)
	for x in range(args.amount):
		vprint(f"\n========= RUN {x+1} =========\n")
		for synthesizer in args.generators:
			vprint(f"\n=== Measurement for synthetic data generator '{synthesizer}' ===\n")
			for label in args.label:
				vprint(f"Starting data set: {label}")
				csv_output = pyRAPL.outputs.CSVOutput(f'measurements/energy_{label}.csv')

				#### Cleaning ####
				if 1 in args.phases:
					if synthesizer == 'benchmark' or x == 0:
						vprint("- Measuring cleaning")
						cleaning_function = getattr(Phase1Cleaning, f"clean_{label}")
						with pyRAPL.Measurement('cleaning', output=csv_output):
							cleaning_function()

				#### Synthetic data generation ####
				if 2 in args.phases:
					if synthesizer != 'benchmark':
						vprint("- Measuring synthetic data generation")
						generator_function = getattr(Phase2SyntheticDataGeneration, synthesizer)
						with pyRAPL.Measurement(f'syn_{synthesizer}', output=csv_output):
							generator_function(label, dict[label])

				#### Preprocessing ####
				if 3 in args.phases:
					preprocessing_function = getattr(Phase3Preprocessing, f"preproc_{label}")
					if synthesizer == 'benchmark':
						vprint("- Measuring preprocessing")
						with pyRAPL.Measurement('preprocessing', output=csv_output):
							preprocessing_function()
						# Check sizes of dataframes
						data = pd.read_csv(f'data/{label}_val_data.csv', nrows=0)
						synt = pd.read_csv(f'data/{label}_train_data.csv')
						for loc, col in enumerate(data.columns):
							if col not in synt.columns:
								synt.insert(loc=loc, column=col, value=0)
						synt.to_csv(f'data/{label}_train_data.csv', index=False)
					else:
						vprint(f"- Measuring preprocessing ({synthesizer})")
						with pyRAPL.Measurement(f'preprocessing_{synthesizer}', output=csv_output):
							preprocessing_function(path="syn/", synthesizer=f"_{synthesizer}")

						# Check sizes of dataframes
						data = pd.read_csv(f'data/{label}_val_data.csv', nrows=0)
						synt = pd.read_csv(f'data/syn/{label}_train_data_{synthesizer}.csv')
						for loc, col in enumerate(data.columns):
							if col not in synt.columns:
								synt.insert(loc=loc, column=col, value=0)
						synt.to_csv(f'data/syn/{label}_train_data_{synthesizer}.csv', index=False)

				#### Machine learning tasks ####
				if 4 in args.phases:
					vprint("- Measuring machine learning task")
					results = pd.DataFrame({'name': [synthesizer]})

					for task in args.tasks:
						task_function = getattr(Phase4MachineLearning, task)

						if synthesizer == 'benchmark':
							vprint(f"  * {task}")
							with pyRAPL.Measurement(f'{task}', output=csv_output):
								acc = task_function(label)
						else:
							vprint(f"  * {task} ({synthesizer})")
							with pyRAPL.Measurement(f'{task}_{synthesizer}', output=csv_output):
								acc = task_function(f"syn/{label}", synthesizer=f"_{synthesizer}")
						results[task] = acc

					# Save accuracies
					results.to_csv(f'measurements/accuracy_{label}.csv', mode='a', index=False,
									header=not os.path.exists(f'measurements/accuracy_{label}.csv'))

				# Save energy measurements
				csv_output.save()
