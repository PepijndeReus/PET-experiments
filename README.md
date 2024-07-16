<h1 align="center">
	Unveiling the Trade-offs
</h1>

<h3>
	Evaluating PETs
</h3>

---

This framework allows for easy evaluation of PETs. PETs are evaluated on energy
and utility metrics. The granulation allows for easy modification and integration
of new PETs and/or data sets.

---

## Installation

First install dependencies
```bash
pip install -r requirements
```

Install `crypten`
```bash
pip install --no-deps crypten
```

## How to?

To download the data sets
```bash
./download_raw
```
To run the experiments
```bash
python3 main.py
```
---

## Structure
- **/raw/**: data in raw format
- **/data/**: cleaned and preprocessed data
- **/dp_ctgan/**: modified version of DP-CTGAN
- **/measurements/**: contains all measurements

## Python files
- **/idle.py**: measures energy consumption while device is idle
- **/model.py**: neural network as machine learning task
- **/main.py**: run the four files below
- **/1_cleaning.py**: remove uneccesary rows and columns
- **/2_synthetic_generation.py**: create the synthetic data
- **/3_preprocess.py**: prepare data for machine learning task
- **/4_train.py**: (encrypted) training on (synthetic) data

---

