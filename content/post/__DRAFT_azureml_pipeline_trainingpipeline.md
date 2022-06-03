---
title: "__DRAFT_azureml_pipeline_trainingpipeline"
date: 2022-05-05
tags: []
draft: true
description: ""
---
## Introduction

![ML Pipeline](/post/img/azureml_pipeline_introduction_pipeline.jpg)
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

## Model training and selection

```python evaluation
'''
Evaluation functions for the model
'''
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate(model, X_val, y_val):
    '''
    Calculates the root of mean squared error of the model
    '''
    # Calculating the root of mean squared error of the model.
    prediction = model.predict(X_val)
    test_mse = mean_squared_error(y_val, prediction)
    sqrt_mse = np.sqrt(test_mse)
    return sqrt_mse

```

```python model training
'''
Train step. Trains various models and selects the best one.
'''
import os
import argparse
import pickle
import pandas as pd
from azureml.core import Run
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from model_evaluation import evaluate

# Getting the run context.
run = Run.get_context()

# Parsing the arguments passed to the script.
parser = argparse.ArgumentParser()
parser.add_argument('--train_y_folder', dest='train_y_folder', required=True)
parser.add_argument('--train_Xt_folder', dest='train_Xt_folder', required=True)
parser.add_argument('--model_folder', dest='model_folder', required=True)

parser.add_argument(
    '--bag_regressor_n_estimators',
    dest='bag_regressor_n_estimators',
    required=True)
parser.add_argument(
    '--decision_tree_max_depth',
    dest='decision_tree_max_depth',
    required=True)
parser.add_argument(
    '--random_forest_n_estimators',
    dest='random_forest_n_estimators',
    required=True)
args = parser.parse_args()

# Reading the data from the `X_train_trans.txt` file and converting it to a numpy array.
X_t_path = os.path.join(args.train_Xt_folder, "data.txt")
X_train_trans = pd.read_csv(X_t_path, header=None).to_numpy()#.squeeze()

# Reading the data from the `y_train.txt` file and converting it to a numpy array.
y_path = os.path.join(args.train_y_folder, "data.txt")
y_train = pd.read_csv(y_path, header=None).to_numpy()#.squeeze()

# Creating an empty dictionary.
results = {}

# Training a linear regression model and evaluating it.
lin_regression = LinearRegression()
lin_regression.fit(X_train_trans, y_train)
sqrt_mse= evaluate(lin_regression, X_train_trans, y_train)
results[lin_regression] = sqrt_mse
run.log('LinearRegression', sqrt_mse)

# Training a bagging regressor model and evaluating it.
bag_regressor = BaggingRegressor(
    n_estimators=int(args.bag_regressor_n_estimators),
    random_state=42)
bag_regressor.fit(X_train_trans, y_train)
sqrt_mse= evaluate(bag_regressor, X_train_trans, y_train)
results[bag_regressor] = sqrt_mse
run.log('BaggingRegressor', sqrt_mse)

# Training a decision tree regressor model and evaluating it.
dec_tree_regressor = DecisionTreeRegressor(
    max_depth=int(args.decision_tree_max_depth))
dec_tree_regressor.fit(X_train_trans, y_train)
sqrt_mse= evaluate(dec_tree_regressor, X_train_trans, y_train)
results[dec_tree_regressor] = sqrt_mse
run.log('DecisionTreeRegressor',sqrt_mse)

# Training a random forest regressor model and evaluating it.
random_forest_regressor = RandomForestRegressor(
    n_estimators=int(args.random_forest_n_estimators),
    random_state=42)
random_forest_regressor.fit(X_train_trans, y_train)
sqrt_mse= evaluate(random_forest_regressor, X_train_trans, y_train)
results[random_forest_regressor] = sqrt_mse
run.log('RandomForestRegressor', sqrt_mse)

# Selecting the model with the lowest RMSE.
selected_model =  min(results, key=results.get)
run.log('selected_model', selected_model)

# Saving the model to a file.
model_path = os.path.join(args.model_folder, 'model.pkl')
with open(model_path, 'wb') as handle:
    pickle.dump(selected_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Uploading the model to the Azure ML workspace and registering it.
run.upload_file(model_path,model_path)
run.register_model(model_path=model_path, model_name='concrete_model')
```

## Deploying the Pipeline

```python pipeline deployment
'''
Execute the training pipeline with the following steps:

- Step 1: Split the data into train and test sets
- Step 2: Prepare the train data for training and generate a prepare data pipeline
- Step 3: Train the model using various algorithms and select the best one
'''
import os
from azureml.pipeline.steps.python_script_step import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.core import Workspace, Experiment, Dataset, Datastore
from azureml.core.runconfig import RunConfiguration
from azureml.core.environment import Environment
from azureml.data.output_dataset_config import OutputFileDatasetConfig
from pytest import param
from azuremlproject import constants

# Loading the workspace from the config file.
config_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    '.azureml')
ws = Workspace.from_config(path=config_path)

# create a new runconfig object
run_config = RunConfiguration()

# Creating a new environment with the name env-7 and 
# installing the packages in the requirements.txt file.
environment = Environment.from_pip_requirements(
    name='env-7',
    file_path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'requirements.txt')
    )
run_config.environment = environment

# Path to the folder containing the scripts that
# will be executed by the pipeline.
source_directory = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'pipeline_steps')

data_store = Datastore(ws, constants.DATASTORE_NAME)

print('Creating train test folders....')

# Creating the output folders for the pipeline.
train_X_folder = OutputFileDatasetConfig(
    destination=(data_store, 'train_X_folder')).register_on_complete('train_X')
train_y_folder = OutputFileDatasetConfig(
    destination=(data_store, 'train_y_folder')).register_on_complete('train_y')

test_X_folder = OutputFileDatasetConfig(
    destination=(data_store, 'test_X_folder')).register_on_complete('test_X')

test_y_folder = OutputFileDatasetConfig(
    destination=(data_store, 'test_y_folder')).register_on_complete('test_y')

train_Xt_folder = OutputFileDatasetConfig(
    destination=(data_store, 'train_Xt_folder')).register_on_complete('train_Xt')

pipeline_folder = OutputFileDatasetConfig(destination=(data_store, 'pipeline_folder'))
model_folder = OutputFileDatasetConfig(destination=(data_store, 'model_folder'))
print('train test folders created')

concrete_dataset = Dataset.get_by_name(ws, 'concrete_baseline')
ds_input = concrete_dataset.as_named_input('concrete_baseline')

print('step_train_test_split ....')
# Splitting the data into train and test sets.
step_train_test_split = PythonScriptStep(
    script_name = 'step_train_test_split.py',
    arguments=[
        '--train_X_folder', train_X_folder,
        '--train_y_folder', train_y_folder,
        '--test_X_folder', test_X_folder,
        '--test_y_folder', test_y_folder],
    inputs = [ds_input],
    compute_target = constants.INSTANCE_NAME,
    source_directory=source_directory,
    allow_reuse = True,
    runconfig = run_config
)
print('step_train_test_split done')

print('step_prepare_data ....')
# step that will execute the script step_prepare_data.py.
step_prepare_data = PythonScriptStep(
    script_name = 'step_prepare_data.py',
    arguments=[
        '--train_X_folder', train_X_folder.as_input('train_X_folder'),
        '--train_Xt_folder', train_Xt_folder,
        '--pipeline_folder', pipeline_folder],
    compute_target = constants.INSTANCE_NAME,
    source_directory=source_directory,
    allow_reuse = True,
    runconfig = run_config
)
print('step_prepare_data done')

print('training_step ....')
# step that will execute the script step_train.py.

param_bag_regressor_n_estimators = PipelineParameter(
    name='The number of estimators of the bagging regressor',
    default_value=20)

param_decision_tree_max_depth = PipelineParameter(
    name='The maximum depth of the decision tree',
    default_value=5)

param_random_forest_n_estimators = PipelineParameter(
    name='The number of estimators of the random forest',
    default_value=5)

training_step = PythonScriptStep(
    script_name = 'step_train.py',
    arguments=[
        '--train_Xt_folder', train_Xt_folder.as_input('train_Xt_folder'),
        '--train_y_folder', train_y_folder.as_input('train_y_folder'),
        '--model_folder', model_folder,
        '--bag_regressor_n_estimators', param_bag_regressor_n_estimators,
        '--decision_tree_max_depth', param_decision_tree_max_depth,
        '--random_forest_n_estimators', param_random_forest_n_estimators],
    compute_target = constants.INSTANCE_NAME,
    source_directory=source_directory,
    allow_reuse = True,
    runconfig=run_config
)
print('training_step done')

# Build the pipeline
print('Building pipeline...')
pipeline = Pipeline(
    workspace=ws,
    steps=[
        step_train_test_split,
        step_prepare_data,
        training_step
    ])
print('Pipeline built')

# Submit the pipeline to be run
experiment = Experiment(ws, 'Concrete_Strength__Pipeline_2')
run = experiment.submit(pipeline)
run.wait_for_completion()
```
