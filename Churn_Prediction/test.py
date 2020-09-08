# This code is from Jupyter Notebook

import numpy as np
import pandas as pd


data_test = pd.read_csv('orange_small_churn_test_data.csv', index_col='ID')

numerical_test = data_test.columns[:190]
categorical_test = data_test.columns[190:]

data_test[numerical_test] = data_test[numerical_test].fillna(0)
data_test[categorical_test] = data_test[categorical_test].fillna('?')

to_drop_from_num_test = []
to_drop_from_cat_test = []

for column in numerical_test:
    d = collections.Counter(data_test[column].values)
    if (d[0] / data_test.shape[0]) >= 0.65:
        to_drop_from_num_test.append(column)
        
for column in categorical_test:
    d = collections.Counter(data_test[column].values)
    if (d['?'] / data_test.shape[0]) >= 0.65:
        to_drop_from_cat_test.append(column)

data_test.drop(to_drop_from_num_test, axis=1, inplace=True)
data_test.drop(to_drop_from_cat_test, axis=1, inplace=True)

numerical_test = []
categorical_test = []
for column in data_test.columns:
    if data_test[column].dtype == 'object':
        categorical_test.append(column)
    else:
        numerical_test.append(column)

# from train.py
lin_depends = ['Var22', 'Var25', 'Var112', 'Var123', 'Var160']

data_test.drop(lin_depends, axis=1, inplace=True)
for feature in lin_depends:
    numerical_test.remove(feature)


# Predictions, estimator from train.py with best parameters
best_predictions_prob = gb_grid_cv.best_estimator_.predict_proba(data_test)


