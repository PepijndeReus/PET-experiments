import argparse
import os
import pyRAPL
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

	parser.add_argument('-d', '--dataset', nargs="*", dest='label',
		choices=['student', 'breast', 'heart', 'census'],
		default=['student', 'breast', 'heart', 'census'])
	parser.add_argument('-s', '--generator', nargs="+", dest='generators',
		choices=['benchmark', 'dpctgan', 'ydata'],
		default=['benchmark'])
	parser.add_argument('-p', '--phases', nargs="*", dest='phases', type=int,
		choices=range(1,5),
		default=range(1,5))
	parser.add_argument('-t', '--ml_task', nargs="+", dest='tasks',
		choices=['knn', 'lr', 'nn'],
		default=['knn', 'lr', 'nn'])
	parser.add_argument('-n', '--amount', type=int, dest='amount', default=1)
	parser.add_argument('--version', action='version', version='%(prog)s 0.1')
	parser.add_argument('-v', '--verbose', action='store_true')


	args = parser.parse_args()

	if not os.path.isdir('raw'):
		print("Data is not present in folder 'raw/'. Downloading the data now!")
		os.system('./download_raw.sh')
	if not all([False for val in [1,2,3,4] if val not in args.phases]):
		print("Warning: not executing all phases. Might result in FileNotFoundError.")
		confirmation = input("Do you want to continue? (y/n) ").lower()
		while confirmation not in ("y", "n"):
		    confirmation = input("Invalid input. Please enter 'y' or 'n': ").lower()
		if confirmation == 'n': exit()
	"""
	if 1 not in args.phases and not os.path.isdir('data'):
		print("Cleaning phase not executed, but 'data/' not present, adding"\
			  " cleaning phase to the pipeline")
		args.phases.append(1)
	if 2 not in args.phases and any(val != 'benchmark' for val in args.generators):
		if not os.path.isdir('data/syn'):
			print("Synthetic data generation phase not chosen, but selected a "\
				  "generator. Adding synthetic data generation to the pipeline")
			args.phases.append(2)
		else: print("Warning: synthesizer chosen, but phase two is not selected.")
	"""
	vprint = print if args.verbose else lambda *a, **k: None

	pyRAPL.setup()
	os.makedirs('measurements', exist_ok=True)

	for _ in range(args.amount):
		vprint(f"\n========= RUN {_} =========\n")
		for synthesizer in args.generators:
			vprint(f"\n=== Measurement for synthetic data generator '{synthesizer}' ===\n")
			for label in args.label:
				vprint(f"Starting data set: {label}")
				csv_output = pyRAPL.outputs.CSVOutput(f'measurements/energy_{label}.csv')

				#### Cleaning ####
				if 1 in args.phases:
					if synthesizer == 'benchmark':
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
							generator_function(label,
								['age', 'class', 'menopause', 'tumor-size',
	                	         'inv-nodes', 'node-caps', 'deg-malig',
	                	         'breast', 'breast-quad', 'irradiat']) #<- TODO dict toevoegen

				#### Preprocessing ####
				if 3 in args.phases:
					preprocessing_function = getattr(Phase3Preprocessing, f"preproc_{label}")
					if synthesizer == 'benchmark':
						vprint("- Measuring preprocessing")
						with pyRAPL.Measurement('preprocessing', output=csv_output):
							preprocessing_function()
					else:
						vprint(f"- Measuring preprocessing ({synthesizer})")
						with pyRAPL.Measurement(f'preprocessing_{synthesizer}', output=csv_output):
							preprocessing_function(path="syn/", synthesizer=f"_{synthesizer}")

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
