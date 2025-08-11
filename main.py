import argparse
import os
import pyRAPL
import yaml
import pandas as pd
import glob

from scripts import Phase1Cleaning
from scripts import Phase2SyntheticDataGeneration
from scripts import Phase3Preprocessing
from scripts import Phase4MachineLearning
from scripts import Phase5Anonymeter

# from codecarbon import OfflineEmissionsTracker
from codecarbon import EmissionsTracker

CODECARBON_OUTPUT = "emissions.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description	= "Measure energy and utility of PETs",
        epilog		= "text ad bottom om help")

    # note: nbsynthetic is not included by default due to failures. more info in our paper
    parser.add_argument('-d', '--dataset', nargs="+", dest='label',
        choices=['student', 'breast', 'heart', 'census'],
        default=['student', 'breast', 'heart', 'census'])
    parser.add_argument('-s', '--generator', nargs="+", dest='generators',
        choices=['benchmark', 'dpctgan', 'ydata', 'ctgan', 'synthcity', 'nbsynthetic', 'datasynthesizer'],
        default=['benchmark', 'dpctgan', 'ydata','ctgan', 'synthcity', 'datasynthesizer'])
    parser.add_argument('-p', '--phases', nargs="+", dest='phases', type=int,
        choices=[1,2,3,4,5],
        default=[1,2,3,4,5])
    parser.add_argument('-t', '--ml_task', nargs="+", dest='tasks',
        choices=['knn', 'lr', 'nn'],
        default=['knn', 'lr', 'nn'])
    parser.add_argument('-n', '--amount', type=int, dest='amount', default=5)
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    parser.add_argument('-c', '--clear', action='store_true', dest='clear', help='Delete CSV files in measurements folder')

    args = parser.parse_args()

    if not os.path.isdir('raw'):
        print("Data is not present in folder 'raw/'. Downloading the data now!")
        os.system('./download_raw.sh')
    """if not all([False for val in [1,2,3,4,5] if val not in args.phases]):
        print("Warning: not executing all phases. Might result in FileNotFoundError.")
        confirmation = input("Do you want to continue? (y/n) ").lower()
        while confirmation not in ("y", "n"):
        confirmation = input("Invalid input. Please enter 'y' or 'n': ").lower()
        if confirmation == 'n': exit()
    """

    vprint = print if args.verbose else lambda *a, **k: None

    os.makedirs('measurements', exist_ok=True)

    # TODO yaml configuratie inladen
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

    # If argument clear is True, clear measurement CSV files
    if args.clear:
        csv_files = glob.glob('measurements/*.csv')
        for file in csv_files:
            try:
                filename = os.path.basename(file)

                # for energy measurements, keep the file but delete all lines after the header
                if filename.startswith("energy_"):
                    # Keep only the first line (header)
                    with open(file, 'r') as f:
                        header = f.readline()
                    with open(file, 'w') as f:
                        f.write(header)
                    print(f"Cleared data from: {file} (kept header)")

                # for accuracy measurement, delete all files.
                else:
                    os.remove(file)
                    print(f"Deleted: {file}")
            
            # error handeling
            except Exception as e:
                print(f"Error processing {file}: {e}")


    print("Given arguments:", args)
    for x in range(args.amount):
        # TODO time total run
        vprint("\n\n=============================\n",
               f"========= RUN {x+1} =========\n",
               "=============================\n\n")
        for synthesizer in args.generators:
            vprint(f"\n\n=== Measurement for synthetic data generator '{synthesizer}' (run {x+1})===\n\n")
            for label in args.label:
                vprint(f"\nStarting data set: {label}")

                if os.path.exists(CODECARBON_OUTPUT):
                    os.remove(CODECARBON_OUTPUT)

                #### Cleaning ####
                if 1 in args.phases:
                    if synthesizer == 'benchmark' or x == 0:
                        vprint("- Measuring cleaning")
                        cleaning_function = getattr(Phase1Cleaning, f"clean_{label}")
                        cleaning_function()

                #### Synthetic data generation ####
                if 2 in args.phases:
                    print("Now running with epsilon = 0.1")
                    if synthesizer != 'benchmark':
                        vprint("- Measuring synthetic data generation")
                        generator_function = getattr(Phase2SyntheticDataGeneration, synthesizer)

                        if synthesizer == 'ydata' or 'ctgan':
                            # takes a lot of time, separate measurement included in source code
                            try:
                                generator_function(label, dict[label])
                            except Exception as e:
                                print(e)
                        else:
                            # with pyRAPL.Measurement(f'syn_{synthesizer}', output=csv_output):
                            # with OfflineEmissionsTracker(country_iso_code="NLD", project_name=f"syn_{synthesizer}") as tracker:
                            with EmissionsTracker(project_name=f"syn_{synthesizer}") as tracker:
                                try:
                                    generator_function(label, dict[label])
                                except Exception as e:
                                    print(e)


                #### Preprocessing ####
                if 3 in args.phases:
                    preprocessing_function = getattr(Phase3Preprocessing, f"preproc_{label}")
                    if synthesizer == 'benchmark':
                        vprint("- Measuring preprocessing")
                        try:
                            preprocessing_function()
                        except Exception as e:
                            print(e)

                        # Check sizes of dataframes
                        data = pd.read_csv(f'data/{label}_val_data.csv', nrows=0)
                        synt = pd.read_csv(f'data/{label}_train_data.csv')
                        for loc, col in enumerate(data.columns):
                            if col not in synt.columns:
                                synt.insert(loc=loc, column=col, value=0)
                        synt.to_csv(f'data/{label}_train_data.csv', index=False)
                    else:
                        vprint(f"- Measuring preprocessing ({synthesizer})")
                        # with pyRAPL.Measurement(f'preprocessing_{synthesizer}', output=csv_output):
                        # with OfflineEmissionsTracker(country_iso_code="NLD", project_name=f"preprocessing_{synthesizer}") as tracker:
                        with EmissionsTracker(project_name=f"preprocessing_{synthesizer}") as tracker:
                            try:
                                preprocessing_function(path="syn/", synthesizer=f"_{synthesizer}")
                            except Exception as e:
                                print(e)

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
                        try:
                            task_function = getattr(Phase4MachineLearning, task)

                            if synthesizer == 'benchmark':
                                vprint(f"  * {task}")
                                # with pyRAPL.Measurement(f'{task}', output=csv_output):
                                # with OfflineEmissionsTracker(country_iso_code="NLD", project_name=f"{task}") as tracker:
                                with EmissionsTracker(project_name=f"{task}") as tracker:
                                    acc = task_function(label)
                            else:
                                vprint(f"  * {task} ({synthesizer})")
                                # with pyRAPL.Measurement(f'{task}_{synthesizer}', output=csv_output):
                                # with OfflineEmissionsTracker(country_iso_code="NLD", project_name=f"{task}_{synthesizer}") as tracker:
                                with EmissionsTracker(project_name=f"{task}_{synthesizer}") as tracker:
                                    acc = task_function(f"syn/{label}", synthesizer=f"_{synthesizer}")
                            results[task] = acc
                        except Exception as e:
                            print(e)

                    # Save accuracies
                    results.to_csv(f'measurements/accuracy_{label}.csv', mode='a', index=False,
                                    header=not os.path.exists(f'measurements/accuracy_{label}.csv'))

    if 5 in args.phases:
        Phase5Anonymeter.measure_anonymity()

    print("End of experiments!")
