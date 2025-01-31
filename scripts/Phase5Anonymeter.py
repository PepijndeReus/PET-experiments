from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
import pandas as pd

original_data = pd.read_csv("data/student_train_data.csv")
control_data = pd.read_csv("data/student_train_labels.csv")

synthesizers = ["ctgan", "datasynthesizer", "dpctgan", "synthcity", "ydata"]

evaluators = {SinglingOutEvaluator: "Singling Out Evaluator",
              LinkabilityEvaluator: "Linkability Evaluator",
              InferenceEvaluator: "Inference Evaluator"}

datasets = ["breast", "census", "heart", "student"]

# Variables for the experiments
n_attacks = 1
confidence_level = 0.95

# =============================================================================

results = []

for synth in synthesizers:
    for evaluator in evaluators:
        for dataset in datasets:

            original_data = pd.read_csv(f"data/{dataset}_train_data.csv")
            control_data = pd.read_csv(f"data/{dataset}_train_labels.csv")
            synthetic_data = pd.read_csv(f"data/syn/{dataset}_train_data_{synth}.csv")

            evaluator_instance = SinglingOutEvaluator(
                ori=original_data,
                syn=synthetic_data,
                control=control_data,
                n_attacks=n_attacks
            )
            evaluator_instance.evaluate()
            res = evaluator_instance.risk(confidence_level=confidence_level)

            results.append([synth, evaluators[evaluator], dataset,
                            res.attack_rate, res.baseline_rate,
                            res.control_rate, res.risk()])

columns = ["Synthesizer", "Evaluator", "Dataset", "Main Attack Rate",
           "Baseline Attack Rate", "Control Attack Rate", "Privacy Risk"]
results_df = pd.DataFrame(results, columns=columns)

results_df.to_csv("anonymeter_results/results.csv", index=False)
