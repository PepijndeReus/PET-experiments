# Install the library. Might need to restart the runtime after installing some dependencies.
#pip install synthcity

import pandas as pd

# stdlib
import sys
import warnings

warnings.filterwarnings("ignore")


# synthcity absolute
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

log.add(sink=sys.stderr, level="INFO")

data = pd.read_csv('/Users/kyradresen/PET-experiments/Data/Student/student-por.csv')


# Define X and y
X = data.drop(columns=["G1"])
y = data["G1"]

# Adding the target to the dataframe (optional, for completeness of the example)
X["target"] = y

# Display X
print(X)

# Preprocessing data with OneHotEncoder or StandardScaler is not needed or recommended. Synthcity handles feature encoding and standardization internally

loader = GenericDataLoader(
    data,
    target_column="G1",
)

# synthcity absolute
from synthcity.plugins import Plugins

Plugins().list()

# synthcity absolute
from synthcity.plugins import Plugins

syn_model = Plugins().get("adsgan")

syn_model.fit(loader)