---
title: "__DRAFT_azureml_pipeline_modeloptimization"
date: 2022-05-05
tags: []
draft: true
description: ""
---
## Introduction

In this second part of this series, we will optimize the model that we created in the [previously](/post/azureml_pipeline_trainingpipeline).

The work described in this section is contained in the following folder structure. In the sections below, we will go through all these files individually.

```text
optimize
│   optimize_model.py
|   requirements.txt
│
└───src
│   │   data_transformer_builder.py
│   │   optimize.py
```

## Model Optimization

```python

'''
Optimization of model parameters
'''
import os
import argparse
import joblib
import pickle
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from azureml.core import Run, Model, Dataset
import pandas as pd

# Parsing the arguments passed to the script.
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, help='Number of estimators')
parser.add_argument('--base_estimator', type=str, help='Base estimator')
parser.add_argument('--max_samples', type=float, help='Max samples')
parser.add_argument('--max_features', type=float, help='Max features')
parser.add_argument('--bootstrap', type=bool, help='Bootstrap')
args = parser.parse_args()
```

``` python
# get the run context and the workspace.
run = Run.get_context()
workspace = run.experiment.workspace

# Get the dataset for X from the workspace and then mount it.
mount_context_X = Dataset.get_by_name(workspace, name='train_X').mount()
mount_context_X.start()
path = mount_context_X.mount_point + '/data.txt'
train_X = pd.read_csv(path, header=None).to_numpy()
# Get the dataset fot y from the workspace and then mount it.
mount_context_y = Dataset.get_by_name(workspace, name='train_y').mount()
mount_context_y.start()
path = mount_context_y.mount_point + '/data.txt'
train_y = pd.read_csv(path, header=None).to_numpy()
```

``` python
# Load the data_transformer from the model store.
data_transformer_path = Model(workspace, 'data_transformer').download(exist_ok = True)
data_transformer = joblib.load(data_transformer_path)

# Load the model from the model store.
model_path = Model(workspace, 'concrete_model').download(exist_ok = True)
model = joblib.load(model_path)
```

``` python
# set the base estimator.
if args.base_estimator == 'LinearRegression':
    model.set_params(base_estimator = LinearRegression())
elif args.base_estimator == 'RandomForestRegressor':
    model.set_params(base_estimator = RandomForestRegressor())
else:
    model.set_params(base_estimator = KNeighborsRegressor())

# Setting the parameters of the model.
model.set_params(n_estimators=args.n_estimators)
# model.set_params(base_estimator=args.base_estimator)
model.set_params(max_samples=args.max_samples)
model.set_params(max_features=args.max_features)
model.set_params(bootstrap=args.bootstrap)
# model.set_params(bootstrap_features=args.bootstrap_features)

# Transforming the test data using the data transformer and
# then predicting the values using the model.
processed_data = data_transformer.transform(train_X)
predictions = model.predict(processed_data)
```

``` python
# Calculating square root of the mean squared error.
test_mse = mean_squared_error(train_y, predictions)
sqrt_mse = np.sqrt(test_mse)
run.log('sqrt_mse', sqrt_mse)

# Saving the model to a file.
model_path = os.path.join('outputs', 'model.pkl')
with open(model_path, 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

mount_context_X.stop()
mount_context_y.stop()

```

### Execution of Optimization Process

```python
'''
Running of experiment_1.py
'''
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.run import PrimaryMetricGoal, HyperDriveRun
from azureml.train.hyperdrive.parameter_expressions import choice
from azuremlproject import constants

# Loading the workspace from the config file.
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.azureml')
w_space = Workspace.from_config(path=config_path)

# Creating a ScriptRunConfig object.
config = ScriptRunConfig(
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'src'),
    script='optimize.py',
    arguments=['--n_estimators', '5'],
    compute_target=constants.TARGET_NAME)
```

``` python
if 'concrete_env' in w_space.environments.keys():
    config.run_config.environment = Environment.get(workspace=w_space, name='concrete_env')
else:
    # Creating an environment from the requirements.txt file.
    environment = Environment.from_pip_requirements(
        name='concrete_env',
        file_path=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'requirements.txt')
        )
    config.run_config.environment = environment
```

``` python
# Creating a dictionary of parameters to be used in the hyperparameter tuning.
param_sampling = RandomParameterSampling( {
    '--base_estimator': choice(
        'LinearRegression', 'RandomForestRegressor', 'KNeighborsRegressor'),
    "--n_estimators": choice(5, 10, 15, 20, 25),
    '--max_samples': choice(0.5, 1.0),
    '--max_features': choice(0.5, 1.0),
    '--bootstrap': choice(1, 0),
    }
)

# Creating a hyperdrive configuration object.
hyperdrive_config = HyperDriveConfig(run_config=config,
                                    hyperparameter_sampling=param_sampling,
                                    primary_metric_name='sqrt_mse',
                                    primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                                    max_total_runs=4,
                                    max_concurrent_runs=4)

# This is submitting the hyperdrive configuration to the experiment.
experiment = Experiment(w_space, 'concrete_tune')
run:HyperDriveRun = experiment.submit(hyperdrive_config )
run.wait_for_completion(show_output=True)

# Checking if the run is completed.
assert run.get_status() == "Completed"
# Printing the status of the run.
print('status', run.get_status())

best_run = run.get_best_run_by_primary_metric()
print('best_run', best_run)
best_run_metrics = best_run.get_metrics()
print('best_run_metrics', best_run_metrics)
parameter_values = best_run.get_details()['runDefinition']['arguments']
print('parameter_values', parameter_values)
best_run.register_model(model_name='final_model', model_path="outputs/model.pkl")

```

## References
