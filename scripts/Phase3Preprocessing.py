"""
This python script will pepare the (synthetic) data for the machine learning
task.

Please note that this data has already been cleaned (see 1_cleaning.py). Data
should be present in the 'data/' folder.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preproc_breast(path="", synthesizer=""):
    '''Preprocesses breast-cancer dataset '''
    breast_train = pd.read_csv(f'data/{path}breast_train{synthesizer}.csv')
    breast_val = pd.read_csv(f'data/breast_val.csv')

    # for col in breast_train:
        # print(col, breast_train[col].unique())

    # see https://github.com/PepijndeReus/PET-experiments/issues/10
    if synthesizer == '_datasynthesizer':
        print("Mapping values from categorical to actual (DataSynthesizer)")
        mappings = {
            'age': {0: '30-39', 1: '40-49', 2: '60-69', 3: '50-59', 4: '70-79'},
            'tumor-size': {0: '0-4', 1: '30-34', 2: '20-24', 3: '15-19', 4: '25-29', 5: '50-54',
                        6: '10-14', 7: '40-44', 8: '35-39', 9: '5-9', 10: '45-49'},
            'inv-nodes': {0: '0-2', 1: '3-5', 2: '6-8', 3: '9-11', 4: '12-14', 5: '15-17'}
        }

        # Function to apply mappings to the DataFrame
        def map_values(df, column_name, mapping_dict):
            return df[column_name].map(mapping_dict).fillna(df[column_name])  # Retain original if no match

        # Apply the mappings to each column in the dataframe
        for column, mapping in mappings.items():
            breast_train[column] = map_values(breast_train, column, mapping)
    
    # for col in breast_train:
        # print(col, breast_train[col].unique())

    def preprocess(data):
        # make binary labels for recurrence
        data['class'] = data['class'].str.replace('no-recurrence-events', '0')
        data['class'] = data['class'].str.replace('recurrence-events', '1')
        data['class'] = data['class'].astype(int)

        # split data with labels
        labels = data['class'].copy()
        data = data.drop(['class'], axis=1)

        # use One-hot encoding for categorial (all) features
        categorial = data.columns.values.tolist()
        data = pd.get_dummies(data, columns = categorial)

        return data, labels

    # apply preprocessing to training and validation set
    train_data, train_labels = preprocess(breast_train)
    val_data, val_labels = preprocess(breast_val)
    # print(train_data.columns)
    # print(val_data.columns)

    # val_data['inv-nodes']== 12-14 missing in val data:
    val_data.insert(
        loc=train_data.columns.get_loc('inv-nodes_12-14'), column='inv-nodes_12-14', value=0)

    # now save the .csv files
    train_data.to_csv(f'data/{path}breast_train_data{synthesizer}.csv', index=False)
    train_labels.to_csv(f'data/{path}breast_train_labels{synthesizer}.csv', index=False)

    val_data.to_csv(f'data/{path}breast_val_data.csv', index=False)
    val_labels.to_csv(f'data/{path}breast_val_labels.csv', index=False)


def preproc_census(path="", synthesizer=""):
    '''Preprocesses census dataset '''
    census_train = pd.read_csv(f'data/{path}census_train{synthesizer}.csv')
    census_val = pd.read_csv(f'data/census_val.csv')

    def preprocess(data):
        # make binary labels for income column
        data['income'] = data['income'].str.replace('<=50K', '0')
        data['income'] = data['income'].str.replace('>50K', '1')
        data['income'] = data['income'].astype(int)

        # make array with labels, remove labels from dataframe
        labels = data['income'].copy()
        data = data.drop(['income'], axis=1)

        # use Min-max scaling for continuous features
        continuous = ['age','capital_gain','capital_loss','hr_per_week']
        data[continuous] = MinMaxScaler().fit_transform(data[continuous])

        # use One-hot encoding for categorial features
        categorial = [feat for feat in data.columns.values if feat not in continuous]
        data = pd.get_dummies(data, columns = categorial)

        return data, labels

    # apply preprocessing to training and validation set
    train_data, train_labels = preprocess(census_train)
    val_data, val_labels = preprocess(census_val)

    # now save the .csv files
    train_data.to_csv(f'data/{path}census_train_data{synthesizer}.csv', index=False)
    train_labels.to_csv(f'data/{path}census_train_labels{synthesizer}.csv', index=False)

    val_data.to_csv(f'data/{path}census_val_data.csv', index=False)
    val_labels.to_csv(f'data/{path}census_val_labels.csv', index=False)

def preproc_student(path="", synthesizer=""):
    '''Preprocesses student dataset '''
    student_train = pd.read_csv(f'data/{path}student_train{synthesizer}.csv')
    student_val = pd.read_csv(f'data/student_val.csv')

    def preprocess(data):
        # convert student grade to pass or fail
        data.loc[data['G3'] < 10, 'G3'] = 0
        data.loc[data['G3'] > 9, 'G3'] = 1
        data['G3'] = data['G3'].astype(int)

        # make array with labels, remove labels from dataframe
        labels = data['G3'].copy()
        data = data.drop(['G3'], axis=1)

        # use Min-max scaling for continuous features
        continuous = ['age','absences','G1','G2']
        data[continuous] = MinMaxScaler().fit_transform(data[continuous])

        # use One-hot encoding for categorial features
        categorial = [feat for feat in data.columns.values if feat not in continuous]
        data = pd.get_dummies(data, columns = categorial)

        return data, labels

    # split into training and validation set, same ratio as Adult set
    train_data, train_labels = preprocess(student_train)
    val_data, val_labels = preprocess(student_val)

    # now save the .csv files
    train_data.to_csv(f'data/{path}student_train_data{synthesizer}.csv', index=False)
    train_labels.to_csv(f'data/{path}student_train_labels{synthesizer}.csv', index=False)

    val_data.to_csv(f'data/{path}student_val_data.csv', index=False)
    val_labels.to_csv(f'data/{path}student_val_labels.csv', index=False)


def preproc_heart(path="", synthesizer=""):
    '''Preprocess heart disease dataset'''
    heart_train = pd.read_csv(f'data/{path}heart_train{synthesizer}.csv')
    heart_val = pd.read_csv(f'data/heart_val.csv')

    def preprocess(data):
        # convert diagnosis to presence or absence
        data.loc[data['num'] > 0, 'num'] = 1
        data.loc[data['num'] == 0, 'num'] = 0

        to_type_int = [col for col in data.columns.values.tolist() if col != 'oldpeak']
        data[to_type_int] = data[to_type_int].astype(int)

        # make array with labels, remove labels from dataframe
        labels = data['num'].copy()
        data = data.drop(['num'], axis=1)

        # use Min-max scaling for continuous features
        continuous = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        data[continuous] = MinMaxScaler().fit_transform(data[continuous])

        # print statement for issue #8:  https://github.com/PepijndeReus/PET-experiments/issues/8
        # print(data['thal'].unique()) # print unique values in col thal

        # use One-hot encoding for categorial features
        categorial = [feat for feat in data.columns.values if feat not in continuous]
        data = pd.get_dummies(data, columns = categorial)

        return data, labels

    # apply preprocessing to training and validation set
    train_data, train_labels = preprocess(heart_train)
    val_data, val_labels = preprocess(heart_val)

    # now save the .csv files
    train_data.to_csv(f'data/{path}heart_train_data{synthesizer}.csv', index=False)
    train_labels.to_csv(f'data/{path}heart_train_labels{synthesizer}.csv', index=False)

    val_data.to_csv(f'data/{path}heart_val_data.csv', index=False)
    val_labels.to_csv(f'data/{path}heart_val_labels.csv', index=False)


if __name__ == "__main__":
    preproc_breast()
    preproc_student()
    preproc_census()
    preproc_heart()
    exit()
    preproc_breastcanc(path="syn/")
    preproc_student(path="syn/")
    #preproc_census(path="syn/")