---
title: "Notes about Azure ML, Part 11 - Model Validation in AzureML"
date: "2023-03-09T13:31:51+01:00"
tags: [machine-learning, azure ml, hyperparameter tuning, model optimization]
draft: false
---
## Introduction

After completing the model selection and optimization phase, the next step is to evaluate the `final_model` using the test dataset. The evaluation process involves executing the full pipeline to transform the test features and predict the corresponding labels. It is expected that the model's performance on the test dataset will be slightly lower than that on the training and validation datasets.

The evaluation process is similar to the previous phases, where the models and test data are loaded, and the pipeline is executed to obtain the test predictors. The obtained predictions are then evaluated using the test labels. This process is executed in the AzureML workspace and triggered by the loader script.

The directory structure for this phase is as follows:

```text
validate
│   validate_model.py
|   requirements.txt
│
└───src
│   │   data_transformer_builder.py
│   │   validate.py
```

The `validate_model.py` script initiates the evaluation process and loads the required dependencies and configurations. The `src` directory contains the `data_transformer_builder.py` script that constructs the data transformer used to transform the test data and the `validate.py` script that executes the pipeline and evaluates the model's performance. The `requirements.txt` file contains the necessary Python package dependencies for the validation process.

### Initialization

The code snippet below sets the stage for this optimization process. First, the necessary libraries are imported: `joblib`, `numpy`, and `pandas` for data manipulation, as well as `sklearn.metrics` for evaluation. Additionally, the code imports the `AzureML` `Run`, `Model`, and `Dataset` classes.

Next, the code initializes the Run context and workspace from Azure ML. The Run context is an important object that tracks information about the model training run, including metrics, output files, and logs. The workspace is the top-level object for working with Azure ML, providing access to the underlying resources like compute clusters and data stores.

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

By retrieving the Run context and workspace, the code enables tracking and management of model training runs in Azure ML. The Run context object can be used to log metrics, upload output files, and retrieve input data. The workspace object, on the other hand, can be used to access and manage resources like compute targets and data stores.

In the next steps, you might perform model training, parameter tuning, and validation with additional code. However, by initializing the Run context and workspace, you're already on your way to managing and optimizing your machine learning models in Azure ML.

### Mounting the datasets

We retrieve and mount two datasets, test_X and test_y, from the Azure ML workspace. The datasets are then loaded into pandas DataFrames and converted to NumPy arrays for use in a machine learning model.

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

To start, the code initializes two mount contexts for the test_X and test_y datasets. The `Dataset.get_by_name()` method retrieves the dataset from the Azure ML workspace by name. Once the dataset is retrieved, the `mount()` method is called on the mount context object, which mounts the dataset to the compute target specified in the workspace configuration.

After the datasets are mounted, the code reads the data files into pandas DataFrames using the `pd.read_csv()` method. The header=None argument specifies that there are no column names in the data files. Finally, the DataFrames are converted to NumPy arrays using the `to_numpy()` method, which returns a NumPy `ndarray` representation of the DataFrame.

### Load the data_transformer from the model store

Next, load a data transformer and a machine learning model from the Azure ML model registry using Python code.

```python
# Load the data_transformer from the model store.
data_transformer_path = Model(workspace, 'data_transformer').download(exist_ok = True)
data_transformer = joblib.load(data_transformer_path)

# Load the model from the model store.
model_path = Model(workspace, 'final_model').download(exist_ok = True)
model = joblib.load(model_path)
```
To start, the code initializes two variables, data_transformer_path and model_path, which point to the data transformer and machine learning model, respectively, in the Azure ML model registry. The `Model()` method retrieves the specified model from the workspace's model registry by name.

Once the models are retrieved, the `download()` method is called on each model object to download the model artifacts to the local filesystem. The `exist_ok` argument specifies that it is okay to download the model even if the local file already exists. The downloaded model artifacts are then loaded into memory using the `joblib.load()` method.

The data transformer is used to preprocess the input data before it is passed to the machine learning model. The machine learning model is the final model trained on the preprocessed data and is used to make predictions on new data.

By loading a data transformer and a machine learning model from the model registry in this way, you can easily deploy models to production environments in Azure ML. Once the models are loaded into memory, they can be used to make predictions on new data using the `predict()` method of the machine learning model.

### Testing

After loading a data transformer and a machine learning model from the Azure ML model registry, the next step is to use the model to make predictions on new data.

```python
# Transforming the test data using the data_transformer and
# then predicting the values using the model.
processed_data = data_transformer.transform(test_X)
predictions = model.predict(processed_data)
```

In the code above, the `test_X` data is transformed using the `data_transformer.transform()` method. This applies the same data preprocessing steps that were applied to the training data when the model was trained. The resulting preprocessed data is then passed to the `model.predict()` method to generate predictions for the test set.

```python
# Calculating square root of the mean squared error.
test_mse = mean_squared_error(test_y, predictions)
sqrt_mse = np.sqrt(test_mse)
run.log('sqrt_mse', sqrt_mse)
```

The code then calculates the mean squared error (MSE) between the actual test_y values and the predicted predictions values using the `mean_squared_error()` method from scikit-learn. The square root of the MSE is then taken and logged to the Azure ML run context using the `run.log()` method. This metric provides a measure of how well the machine learning model is performing on the test set.

```python
mount_context_X.stop()
mount_context_y.stop()
```

Finally, the code unmounts the test data used for prediction.

### Execution

 The azureml Python library provides a convenient way to run experiments and validate the models within the Azure Machine Learning environment.

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
```

The code above loads the Azure Machine Learning workspace from a configuration file. This file specifies the credentials and other settings needed to access the workspace.

```python
# Create a ScriptRunConfig object.
config = ScriptRunConfig(
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'src'),
    script='validate.py',
    arguments=[],
    compute_target=constants.INSTANCE_NAME)
```

The `ScriptRunConfig` object is used to define the script to be executed and the environment it will be executed in. In this code, the `source_directory` points to the directory containing the Python script to be executed, and script specifies the name of the script to run. The `arguments` parameter can be used to pass arguments to the script, and `compute_target` specifies the name of the compute target to use for the experiment.

```python
experiment = Experiment(w_space, 'concrete_validate')
run = experiment.submit(config)
run.wait_for_completion(show_output=True)

# Check if the run is completed.
assert run.get_status() == "Completed"
# Print the status of the run.
print('status', run.get_status())
```

Finally, the `Experiment` object is used to submit the experiment for execution. In this case, the experiment is given a name and associated with the specified workspace. The `submit()` method is used to start the experiment with the specified configuration.

By using azureml to define and run experiments, you can automate and manage the machine learning development and validation process within the Azure Machine Learning environment.