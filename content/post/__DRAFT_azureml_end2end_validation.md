---
title: "Model Validation"
date: 2022-05-05
tags: []
draft: true
description: ""
---

## Introduction

After working with models, where we selected the stsem that worked reasonably well and optimized it as mich as possible, It is now time to evaluate our **final_model** using the test data set. This phase is pretty straight forward, we just run the full pipeline so that we transform the test features and then we predict the test labels.

The expected performance should be slightly inferior that that obtained in the previous post.

The validation process i very similar to the other processes that we discussed in this exercise; a validation process will load the models and the test data, execute the pipeline to obtain the test predictors and then use the test labels to evaluate the model. This process is executed in our AzureMl workspace and is uploaded and initiated by loader script.

The structure of the files used in this pase is therefore as follows:

```text
validate
│   validate_model.py
|   requirements.txt
│
└───src
│   │   data_transformer_builder.py
│   │   validate.py
```

In the next section we will looks at our scripts in some more detail.

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
```

```python
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
```

```python
# Load the data_transformer from the model store.
data_transformer_path = Model(workspace, 'data_transformer').download(exist_ok = True)
data_transformer = joblib.load(data_transformer_path)

# Load the model from the model store.
model_path = Model(workspace, 'final_model').download(exist_ok = True)
model = joblib.load(model_path)
```

```python
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

### execution script

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
```

```python
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
```

```python

experiment = Experiment(w_space, 'concrete_validate')
run = experiment.submit(config)
run.wait_for_completion(show_output=True)

# Check if the run is completed.
assert run.get_status() == "Completed"
# Print the status of the run.
print('status', run.get_status())
```
