import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

train_df = pd.read_csv('Training.csv')
test_df = pd.read_csv('Testing.csv')
#train_df = train_df.drop('Unnamed: 133', axis=1)
def data_cleanup(df):
    '''
    df: pandas dataframe
    '''
    if type(df)!=pd.core.frame.DataFrame:
        raise ValueError('input is not a pandas dataframe')
    working_df = df.copy()
    cols = working_df.columns
    converted_columns = {}
    for col in cols:
        if working_df[col].dtype == 'O':
            unique_values = working_df[col].unique()
            converted_values = {v:k for k,v in enumerate(unique_values)}
            for value in unique_values:
                working_df[col] = working_df[col].replace(value, converted_values[value])
            converted_columns[col] = converted_values
    return working_df, converted_columns
cleaned_train_df, cleaned_train_df_index = data_cleanup(train_df)
cleaned_test_df = test_df.replace(cleaned_train_df_index)
print("train_df.coloumns==",cleaned_train_df.columns)

X_train = cleaned_train_df[[*cleaned_train_df][:-1]].to_numpy()
y_train = np.array(cleaned_train_df[[*cleaned_train_df][-1]])

X_test = cleaned_test_df[[*cleaned_test_df][:-1]].to_numpy()

y_test = np.array(cleaned_test_df[[*cleaned_test_df][-1]])
model = KNeighborsClassifier()
k_neighbours = len(cleaned_train_df.prognosis.unique())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

