import pandas as pd

pd.set_option('display.max_rows', 10000)
df = pd.read_csv('data/syn/breast_train.csv')

print(df['deg-malig'].describe())
print(df['deg-malig'] == 0)
