---
title: "__DRAFT_azureml_pipeline_trainingpipeline"
date: 2022-05-05
tags: []
draft: true
description: ""
---


## Create Train and Test sets

```python train_test_split

'''
Split Train / Test Pipeline Step
'''
import os
import argparse
import numpy as np
from azureml.core import Run
from sklearn.model_selection import StratifiedShuffleSplit

RANDOM_STATE = 42

# Parsing the arguments passed to the script.
parser = argparse.ArgumentParser()
parser.add_argument('--train_X_folder', dest='train_X_folder', required=True)
parser.add_argument('--train_y_folder', dest='train_y_folder', required=True)
parser.add_argument('--test_X_folder', dest='test_X_folder', required=True)
parser.add_argument('--test_y_folder', dest='test_y_folder', required=True)
args = parser.parse_args()

# This is the code that is used to read the data from the dataset 
# that was created in the previous step.
concrete_dataset = Run.get_context().input_datasets['concrete_baseline']
df_concrete = concrete_dataset.to_pandas_dataframe()

# prepare the data, we will use a stratified split to create a train and test set
# first we split the data into 5 buckets according to the cement content
df_concrete['cement_cat'] = np.ceil((df_concrete['cement'] / 540) * 5)

train_set, test_set =  None, None

# we use a stratified split to create a train and test set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
for train_idx, test_idx in split.split(df_concrete, df_concrete['cement_cat']):
    train_set = df_concrete.loc[train_idx]
    test_set = df_concrete.loc[test_idx]

# remove cement_cat in train and test datasets
for set_ in (train_set, test_set):
    set_.drop('cement_cat', axis=1, inplace=True)

# This is splitting the data into training and test sets.
X_train = train_set.drop('strength', axis=1)
y_train = train_set['strength'].copy()
X_test = test_set.drop('strength', axis=1)
y_test = test_set['strength'].copy()

# Write outputs
np.savetxt(os.path.join(args.train_X_folder, "data.txt"), X_test, delimiter=",")
np.savetxt(os.path.join(args.train_y_folder, "data.txt"), y_test, delimiter=",")

np.savetxt(os.path.join(args.test_X_folder, "data.txt"), X_test, delimiter=",")
np.savetxt(os.path.join(args.test_y_folder, "data.txt"), y_test, delimiter=",")

# This is logging the shape of the data.
run = Run.get_context()
run.log('X_train', X_train.shape)
run.log('y_train', y_train.shape)
run.log('X_test', X_test.shape)
run.log('y_test', y_test.shape)

```


## Preparing data for training

```python data_processor
'''
Data Preprocessor

- combines the fine aggregate and coarse aggregate into a single feature
- scales the data using StandardScaler
'''
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

coarse_agg_ix, fine_agg_ix = 5, 6

class CombinedAggregateAdder(BaseEstimator, TransformerMixin):
    '''
    sums the aggregate features in the dataset
    '''
    def __init__(self):
        '''
        initializes the class
        '''
        pass

    def fit(self, X, y=None):
        '''
        fit function. returns self
        '''
        return self

    def transform(self, x_val, y_val=None):
        '''
        Transformation function
        '''
        aggregate_total = x_val[:, coarse_agg_ix] + x_val[:, fine_agg_ix]
        x_val = np.delete(x_val, [coarse_agg_ix, fine_agg_ix], axis=1)
        return np.c_[x_val, aggregate_total]

def transformation_pipeline():
    '''
    Pipeline Creator
    '''
    # Creating a pipeline that will first add the
    # two aggregate features and then scale the data.
    pipeline = Pipeline([
        ('CombinedAggregateAdder', CombinedAggregateAdder()),
        ('StandardScaler', StandardScaler())
    ])

    return pipeline

def pipeline_fit_transform_save(data):
    '''
    Creates a transformation pipeline using data.
    Returns the transformed data and the pipeline.
    '''
    pipeline = transformation_pipeline()
    data_transformed = pipeline.fit_transform(data)

    return pipeline, data_transformed
```

```python data_processor
'''
Data Preprocessor step of the pipeline.
'''
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from azureml.core import Run
from data_preprocessor import pipeline_fit_transform_save

# The dataset is specified at the pipeline definition level.
RANDOM_STATE = 42

run = Run.get_context()

# Parsing the arguments passed to the script.
parser = argparse.ArgumentParser()
parser.add_argument('--train_X_folder', dest='train_X_folder', required=True)
parser.add_argument('--train_Xt_folder', dest='train_Xt_folder', required=True)
parser.add_argument('--pipeline_folder', dest='pipeline_folder', required=True)
args = parser.parse_args()

# Loading the data from the data store.
X_train_path = os.path.join(args.train_X_folder, "data.txt")

X_train = pd.read_csv(X_train_path, header=None).to_numpy()#.squeeze()


# X_train = np.loadtxt(X_train_path, delimiter=",")
run.log('X_train', X_train.shape)

# Fitting the pipeline to the training data and transforming the training data.
pipeline, X_train_transformed = pipeline_fit_transform_save(X_train)
run.log('X_train_transf', X_train_transformed.shape)
run.log('pipeline', pipeline)

# Saving the transformed data to the data store.
np.savetxt(
    os.path.join(args.train_Xt_folder, "data.txt"),
    X_train_transformed,
    delimiter=",",
    fmt='%s')

# Creating a dataframe with the column names and the transformed data.
column_names = ['cement', 'slag', 'ash', 'water', 'superplastic', 'totalagg', 'age']
Xt_train = pd.DataFrame(X_train_transformed, columns=column_names)

# Saving the pipeline to a file.
pipeline_path = os.path.join(args.pipeline_folder, 'data_pipeline.pkl')
with open(pipeline_path, 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Registering the model in the Azure ML workspace.
run.upload_file(pipeline_path,pipeline_path)
run.register_model(model_path=pipeline_path, model_name='data_pipeline')
```
