from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
import pandas as pd
from copy import deepcopy

synthesizer_names = ["ctgan", "datasynthesizer", "dpctgan", "synthcity", "ydata"]

# evaluators = {SinglingOutEvaluator: "Singling Out Evaluator",
#               LinkabilityEvaluator: "Linkability Evaluator",
#               InferenceEvaluator: "Inference Evaluator"}

evaluators = {SinglingOutEvaluator: "Singling Out Evaluator"}

float_columns = {
    "breast": [],
    "census": [0, 1, 2, 3],
    "heart": [0, 1, 2, 3, 4],
    "student": [0, 1, 2, 3]
}

# =============================================================================
import logging

# create a file handler
file_handler = logging.FileHandler("anonymeter.log")

# set the logging level for the file handler
file_handler.setLevel(logging.DEBUG)

# add the file handler to the logger
logger = logging.getLogger("anonymeter")
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)
# =============================================================================

def measure_anonymity(n_attacks = 1, confidence_level = 0.95, datasets = ["breast", "census", "heart", "student"]):

    print("DEBUG: Entered function")

    datasets = [datasets[0]]

    # global synthesizer_names
    # synthesizer_names = [synthesizer_names[0]]

    measurement_results = []

    for synthesizer_name in synthesizer_names:
        for evaluator_class in evaluators:
            for dataset in datasets:

                print(f"DEBUG: ITERATION: {synthesizer_name = }, evaluator = {evaluators[evaluator_class]}, {dataset = }")

                original_data = pd.read_csv(f"data/{dataset}_train_data.csv")
                synthetic_data = pd.read_csv(f"data/syn/{dataset}_train_data_{synthesizer_name}.csv")
                control_data = pd.read_csv(f"data/{dataset}_val_data.csv")

                for data_df in [original_data, synthetic_data, control_data]:
                    columns = data_df.columns
                    columns = [col.replace('-', '_') for col in columns]
                    columns = [col.replace('(', '_') for col in columns]
                    columns = [col.replace(')', '_') for col in columns]
                    columns = [col.replace('&', '_') for col in columns]
                    data_df.columns = columns

                for data_df in [original_data, synthetic_data, control_data]:
                    for i in range(len(data_df.columns)):
                        if i in float_columns[dataset]:
                            data_df.iloc[:, i] = data_df.iloc[:, i].astype(float)
                        else:
                            data_df.iloc[:, i] = data_df.iloc[:, i].astype(bool)

                evaluator_instance = evaluator_class(
                    ori=original_data,
                    syn=synthetic_data,
                    control=control_data,
                    n_attacks=n_attacks
                )
                try:
                    evaluator_instance.evaluate()
                except RuntimeError:
                    print(f"RuntimeError occured in run: {synthesizer_name=}, {evaluators[evaluator_class]=}, {dataset=}")
                    continue
                evaluator_instance.risk(confidence_level=confidence_level)

                res = evaluator_instance.results()

                results_temp = [synthesizer_name, evaluators[evaluator_class],
                                dataset, res.attack_rate, res.baseline_rate,
                                res.control_rate, res.risk()]

                measurement_results.append(deepcopy(results_temp))

    columns = ["Synthesizer", "Evaluator", "Dataset", "Main Attack Rate",
               "Baseline Attack Rate", "Control Attack Rate", "Privacy Risk"]
    results_df = pd.DataFrame(measurement_results, columns=columns)

    results_df.to_csv(f"anonymeter_results/results_{datasets[0]}_{n_attacks}.csv", index=False)
