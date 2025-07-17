from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
import pandas as pd
from copy import deepcopy
import random

synthesizer_names = ["ctgan", "datasynthesizer", "dpctgan", "synthcity", "ydata"]

evaluators = {SinglingOutEvaluator: "Singling Out Evaluator",
              LinkabilityEvaluator: "Linkability Evaluator",
              InferenceEvaluator: "Inference Evaluator"}

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

    measurement_results = []

    for synthesizer_name in synthesizer_names:
        for dataset in datasets:

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

            """
            KNOWLEDGE ATTACKS:
            """

            # Singling Out Evaluator

            SO_evaluator = SinglingOutEvaluator(
                ori=original_data,
                syn=synthetic_data,
                n_attacks=n_attacks,
                n_cols=3,
                control=control_data,
                max_attempts=10_000_000
            )

            SO_evaluator.evaluate(mode="multivariate")
            SO_risk, _ = SO_evaluator.risk(confidence_level=confidence_level,
                                           baseline=False)
            SO_res = SO_evaluator.results(confidence_level=confidence_level)

            results_temp = ["knowledge", synthesizer_name, "Singling Out Evaluator", dataset,
                            SO_res.attack_rate, SO_res.baseline_rate,
                            SO_res.control_rate, SO_risk]

            measurement_results.append(deepcopy(results_temp))

            # Linkability Evaluator

            if dataset == 'census':
                """
                Note, this contains the definitions for the linkability links as defined in
                https://github.com/statice/anonymeter/blob/main/notebooks/anonymeter_example.ipynb

                The first list contains the source data (e.g. medical or school results),
                whereas the second list contains general data (e.g. age and sex).
                We applied these categorisations over each dataset below.
                """

                aux_cols = [
                ['type_employer', 'education', 'hr_per_week', 'capital_loss', 'capital_gain'],
                [ 'race', 'sex', 'fnlwgt', 'age', 'country']
                ]
            elif dataset == 'heart':
                aux_cols = [
                ['cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num'],
                [ 'age', 'sex']
                ]
            elif dataset == 'student':
                aux_cols = [
                ['famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3'],
                [ 'school', 'sex', 'age', 'address']
                ]
            elif dataset == 'breast':
                aux_cols = [
                ['class,menopause,tumor-size,inv-nodes,node-caps,deg-malig,breast,breast-quad,irradiat'],
                [ 'age']
                ]

            LI_evaluator = LinkabilityEvaluator(
                ori=original_data,
                syn=synthetic_data,
                aux_cols=aux_cols,
                n_attacks=n_attacks,
                n_neighbors=3,
                control=control_data
            )

            LI_evaluator.evaluate()
            LI_risk, _ = LI_evaluator.risk(confidence_level=confidence_level,
                                           baseline=False,
                                           n_neighbors=None)
            LI_res = LI_evaluator.results(confidence_level=confidence_level)

            results_temp = ["knowledge", synthesizer_name, "Linkability Evaluator", dataset,
                            LI_res.attack_rate, LI_res.baseline_rate,
                            LI_res.control_rate, LI_risk]
            measurement_results.append(deepcopy(results_temp))

            # Inference Evaluator
            """
            Age is a common denominator in many datasets.
            Therefore we pick age as it exhibits the largest risk and thereby expose a lot of information on the original data.
            """

            IN_evaluator = InferenceEvaluator(
                ori=original_data,
                syn=synthetic_data,
                control=control_data,
                aux_cols=columns,
                secret='age',
                regression=None,
                n_attacks=n_attacks
            )

            IN_evaluator.evaluate()
            IN_risk, _ = IN_evaluator.risk(confidence_level=confidence_level,
                                           baseline=False)
            IN_res = IN_evaluator.results(confidence_level=confidence_level)

            results_temp = ["knowledge", synthesizer_name, "Inference Evaluator", dataset,
                            IN_res.attack_rate, IN_res.baseline_rate,
                            IN_res.control_rate, IN_risk]
            measurement_results.append(deepcopy(results_temp))

            """
            STOCHASTIC ATTACKS:
            """

            # Singling Out Evaluator

            SO_evaluator = SinglingOutEvaluator(
                ori=original_data,
                syn=synthetic_data,
                n_attacks=n_attacks,
                n_cols=3,
                control=control_data,
                max_attempts=10_000_000
            )

            SO_evaluator.evaluate(mode="multivariate")
            SO_risk, _ = SO_evaluator.risk(confidence_level=confidence_level,
                                           baseline=False)
            SO_res = SO_evaluator.results(confidence_level=confidence_level)

            results_temp = ["stochastic", synthesizer_name, "Singling Out Evaluator", dataset,
                            SO_res.attack_rate, SO_res.baseline_rate,
                            SO_res.control_rate, SO_risk]

            measurement_results.append(deepcopy(results_temp))

            # Linkability Evaluator

            all_cols = list(original_data.columns)

            left = random.sample(all_cols, len(all_cols) // 2)
            right = [col for col in all_cols if col not in left]

            aux_cols = [left, right]

            LI_evaluator = LinkabilityEvaluator(
                ori=original_data,
                syn=synthetic_data,
                aux_cols=aux_cols,
                n_attacks=n_attacks,
                n_neighbors=3,
                control=control_data
            )

            LI_evaluator.evaluate()
            LI_risk, _ = LI_evaluator.risk(confidence_level=confidence_level,
                                           baseline=False,
                                           n_neighbors=None)
            LI_res = LI_evaluator.results(confidence_level=confidence_level)

            results_temp = ["stochastic", synthesizer_name, "Linkability Evaluator", dataset,
                            LI_res.attack_rate, LI_res.baseline_rate,
                            LI_res.control_rate, LI_risk]
            measurement_results.append(deepcopy(results_temp))

            # Inference Evaluator

            all_cols = list(original_data.columns)

            secret = random.choice(all_cols)
            aux_cols = [col for col in columns if col != secret]

            IN_evaluator = InferenceEvaluator(
                ori=original_data,
                syn=synthetic_data,
                control=control_data,
                aux_cols=aux_cols,
                secret=secret,
                regression=None,
                n_attacks=n_attacks
            )

            IN_evaluator.evaluate()
            IN_risk, _ = IN_evaluator.risk(confidence_level=confidence_level,
                                           baseline=False)
            IN_res = IN_evaluator.results(confidence_level=confidence_level)

            results_temp = ["stochastic", synthesizer_name, "Inference Evaluator", dataset,
                            IN_res.attack_rate, IN_res.baseline_rate,
                            IN_res.control_rate, IN_risk]
            measurement_results.append(deepcopy(results_temp))

    columns = ["attack_type", "Synthesizer", "Evaluator", "Dataset", "Main Attack Rate",
               "Baseline Attack Rate", "Control Attack Rate", "Privacy Risk"]
    results_df = pd.DataFrame(measurement_results, columns=columns)

    results_df.to_csv(f"anonymeter_results/results_{n_attacks}.csv", index=False)

