# This code is from Jupyter Notebook

import random
import imblearn
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import category_encoders as ce

from sklearn import model_selection, linear_model, ensemble, metrics, pipeline, preprocessing
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks

#Load data
data = pd.read_csv('orange_small_churn_train_data.csv', index_col='ID')

data_train = data.drop('labels', axis=1).iloc[:-1, :]
labels_train = data['labels'][:-1]

# Separate real and categorical signs
numerical = data_train.columns[:190]
categorical = data_train.columns[190:]

# Missing values ​​in real signs are replaced by 0, in categorical ones - with a question mark
data_train[numerical] = data_train[numerical].fillna(0)
data_train[categorical] = data_train[categorical].fillna('?')

# Exclude features with a high percentage of sparsity (more than 40%) from the dataset, they will not bring much benefit for training
to_drop_from_num = []
to_drop_from_cat = []

for column in numerical:
    d = collections.Counter(data_train[column].values)
    if (d[0] / data_train.shape[0]) >= 0.65:
        to_drop_from_num.append(column)
        
for column in categorical:
    d = collections.Counter(data_train[column].values)
    if (d['?'] / data_train.shape[0]) >= 0.65:
        to_drop_from_cat.append(column)

# Exclude the selected noise material and categorical features from the data
data_train.drop(to_drop_from_num, axis=1, inplace=True)
data_train.drop(to_drop_from_cat, axis=1, inplace=True)

# Update feature lists
numerical = []
categorical = []
for column in data_train.columns:
    if data_train[column].dtype == 'object':
        categorical.append(column)
    else:
        numerical.append(column)

# Check if there are any linearly dependent numerical features
corr_matrix_num = data_train[numerical].corr()
sns.heatmap((corr_matrix_num > 0.85) &  (corr_matrix_num < 0.90))

lin_depends = ['Var22', 'Var25', 'Var112', 'Var123', 'Var160']

# Leave only one of the features and update the list of features
data_train.drop(lin_depends, axis=1, inplace=True)
for feature in lin_depends:
    numerical.remove(feature)

# Considering that among categorical features there are features with a large range of values, we divide all categorical features into two groups: less than 40 unique values ​​and more
categorical_less_40 = []
categorical_above_40 = []
for column in categorical:
    if len(data_train[column].unique()) < 40:
        categorical_less_40.append(column)
    else:
        categorical_above_40.append(column)

# Get lists of real indexes and categorical features for the pipeline
numerical_indices = np.array([(column in numerical) for column in data_train.columns])
categorical_indices = np.array([(column in categorical) for column in data_train.columns])
categorical_indices_less_40 = np.array([(column in categorical_less_40) for column in data_train.columns])
categorical_indices_above_40 = np.array([(column in categorical_above_40) for column in data_train.columns])

# Choose gradient boosting as a classifier model
gb_classifier = ensemble.GradientBoostingClassifier(random_state=14)


# Pipeline
estimator = imblearn.pipeline.Pipeline(steps = [
    ('feature_processing', pipeline.FeatureUnion(transformer_list = [        
            #numeric
            ('numeric_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, numerical_indices])),
                ('scaling', preprocessing.StandardScaler(with_mean = 0.5, with_std = 1))            
                        ])),
        
            #categorical_less_40
            ('categorical_less_40_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, categorical_indices_less_40])),
                ('one-hot_encoding', ce.OrdinalEncoder())
                        ])),
        
                    #categorical_above_40
            ('categorical_above_40_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, categorical_indices_above_40])),
                ('loo_encoding', ce.CatBoostEncoder())
                        ])),
        ])),
    ('under_sampling', TomekLinks(sampling_strategy='majority' , n_jobs=-1)),
    ('model_fitting', gb_classifier)
    ]
)

# Train
parameters_grid = {
    'feature_processing__numeric_variables_processing__scaling__with_mean' : [0.5],
    'model_fitting__n_estimators' : [450, 500, 600, 700, 800],
    'model_fitting__learning_rate' : [0.02],
}

gb_grid_cv = model_selection.GridSearchCV(estimator, parameters_grid, scoring = 'roc_auc', cv = 3, n_jobs=-1)

# Print best score
print(gb_grid_cv.best_score_)
print(gb_grid_cv.best_params_)

