---
title: "Model Validation"
date: 2022-05-05
tags: []
draft: true
description: ""
---

## Introduction

In this second part of this series of posts, we will optimize the model we created in the [previously](/post/azureml_pipeline_trainingpipeline) by selecting the best set of hyperparameters, or model configuration parameters, that affect the training process. Hyperparameters differ from model parameters in that they are not learnt through some automated process but are chosen by the data scientist. In general, we cannot use techniques to understand model parameters, such as gradient descent, to learn hyperparameters, although they ultimately affect the loss function as well.

The problem of hyperparameter optimization is therefore in finding the optimal model in an $n$ dimensional space; where $n$ is the number of hyperparameters that are being optimized. We refer to this $n$ dimensional space as the **search space**.

The work described in this section is contained in the following folder structure. In the sections below, we will go through all these files individually.

```text
validate
│   validate_model.py
|   requirements.txt
│
└───src
│   │   data_transformer_builder.py
│   │   validate.py
```



```python

'''
Optimization of model parameters
'''
import joblib
import numpy as np

from sklearn.metrics import mean_squared_error
from azureml.core import Run, Model, Dataset
import pandas as pd

# get the run context and the workspace.
run = Run.get_context()
workspace = run.experiment.workspace

# Get the dataset for X from the workspace and then mount it.
mount_context_X = Dataset.get_by_name(workspace, name='test_X').mount()
mount_context_X.start()
path = mount_context_X.mount_point + '/data.txt'
test_X = pd.read_csv(path, header=None).to_numpy()

# Get the dataset fot y from the workspace and then mount it.
mount_context_y = Dataset.get_by_name(workspace, name='test_y').mount()
mount_context_y.start()
path = mount_context_y.mount_point + '/data.txt'
test_y = pd.read_csv(path, header=None).to_numpy()

run.log('test_X', test_X.shape)

# Load the data_transformer from the model store.
data_transformer_path = Model(workspace, 'data_transformer').download(exist_ok = True)
data_transformer = joblib.load(data_transformer_path)

# Load the model from the model store.
model_path = Model(workspace, 'final_model').download(exist_ok = True)
model = joblib.load(model_path)

# Transforming the test data using the data_transformer and
# then predicting the values using the model.
processed_data = data_transformer.transform(test_X)
predictions = model.predict(processed_data)

# Calculating square root of the mean squared error.
test_mse = mean_squared_error(test_y, predictions)
sqrt_mse = np.sqrt(test_mse)
run.log('sqrt_mse', sqrt_mse)

mount_context_X.stop()
mount_context_y.stop()
```


```python
'''
Running the model on the test data
'''
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azuremlproject import constants

# Load the workspace from the config file.
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.azureml')
w_space = Workspace.from_config(path=config_path)

# Create a ScriptRunConfig object.
config = ScriptRunConfig(
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'src'),
    script='validate.py',
    arguments=[],
    compute_target=constants.INSTANCE_NAME)

if 'concrete_env' in w_space.environments.keys():
    config.run_config.environment = Environment.get(workspace=w_space, name='concrete_env')
else:
    # Create an environment from the requirements.txt file.
    environment = Environment.from_pip_requirements(
        name='concrete_env',
        file_path=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'requirements.txt')
        )
    config.run_config.environment = environment


experiment = Experiment(w_space, 'concrete_validate')
run = experiment.submit(config)
run.wait_for_completion(show_output=True)

# Check if the run is completed.
assert run.get_status() == "Completed"
# Print the status of the run.
print('status', run.get_status())
```