---
title: "Azureml_pipeline_introduction"
date: 2022-05-15
tags: []
draft: true
description: ""
---

## Introduction

In the previous posts in this series we have examined some of the various features of [Azure Machine Learning](/tags/azure-ml). We have executed some experiments and have seen the results. We will now try to look into a more complete, end to end, Machine Learning project in Azure, that will utilize leverage the power of other Azure ML features such as:

- **Azure Machine Learning Pipelines**. Azure ML pipelines a way whereby discrete steps (or subtasks) are executed as a workflow. It is important to notice that the concept of pipeline is recurrent in many Azure services. In particular, Microsoft suggest that Azure ML pipelines should be used for Model Orchestration, that a process that produces a model from data. Another different pipeline type in Azure is that used for data processing; where data produces a new data set. In this case, Microsoft suggest products Azure Data Factory or Azure Data Bricks. Nevertheless, it is possible to use Azure ML pipelines to prepare data, if the data transformation is not too complex. Other tasks that can be performed with Azure ML pipelines include:
  - Training configuration
  - Training and validation
  - Repeatable deployments

- **Azure Machine Learning Hyperparameter Tuning**. We will use the HyperDrive package to tune the hyperparameters of our machine learning model.

- **Model Deployment into Azure**. We will deploy our model into Azure as a web service.

### A disclaimer

This collection of posts is not intended to be a complete guide to Azure Machine Learning model building and deployment or an guide towards building machine learning pipelines. It is intended to be a starting point for those who are interested in using Azure Machine Learning, and to introduce some of the interesting features available in the platform.

## Structure of these posts

This project has been divided into a number of smaller posts so to limit the length and content of each post. It consists of the following entries:

1. [An introduction to the project, considerations and data uploading to Azure](/post/azureml_pipeline_introduction)
2. [Pipeline Creation](/post/azureml_pipeline_creation)
3. [Hyperparameter Tuning](/post/azureml_pipeline_hyperparameter_tuning)
4. [Model Deployment](/post/azureml_pipeline_model_deployment)

## AzureML development considerations and project structure

Following a [post in stackoverflow,](https://stackoverflow.com/questions/6323860/sibling-package-imports) that discussed methods to eliminate sibling import problems, I have decided to create a consistent project structure for my python projects, thus making all my development easier and consistent. I have also created a template for the project structure, which can be found in the [this repository](https://github.com/carmelgafa/python_environment_creator). In this particular exercise, I have names the project as **azuremlproject**.

At a high level, the project structure is as follows:

- **.azuremlproject**  -  This folder contains the Azure Machine Learning files.
  - **.azureml**  -  This folder contains the Azure Machine Learning files.
  - **data**  -  This folder contains the data that will be used in the experiments.
  - **docs**  -  This folder contains the documentation for the project.
  - **experiments**  -  This folder contains the experiments.
    - **experiment_1**  -  This folder contains the experiment 1.
    - **$\dots$**
    - **experiment_14**  -  This folder contains the experiment 14.
      - **deply**  -  This folder contains the deployment of the model.
      - **optimize**  -  This folder contains the hyperparameter optimization.
      - **train**  -  This folder contains the training of the model.
      - **upload**  -  This folder contains the upload of the data.
      - **validate**  -  This folder contains the validation of the model.
    - **$\dots$**
  - **workspace_creation**

As discussed in the [previous posts in this site](https://carmelgafa.com/tags/azure-ml/), I have created a set scripts that create the Azure resource group and AzureML Workspace, with its Compute resources and Azure ML Datastore. This makes it easier to just delete the resource group and recreate it whenever necessary, thus saving some money. The workspace creation scripts are driven from a parameters file that contains the names of the various entities that will be created. The azureml workspace configuration is then stored in the file **.azureml/config.json**.

## Our Project

In this exercise, we will use the [UCI Concrete Compressive Strength Dataset](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength), that is both straightforward and easy to use. It is also a small dataset, and therefore manipulation and exploration of the data is normally carried out quickly.

The data set contains the following features:

| Feature | Description    | Units | Type |
|:--------|:--------------:| ----- | ---- |
|Cement (component 1)&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; quantitative&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  | kg in a m3 mixture &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Input Variable|
|Blast Furnace Slag (component 2) | quantitative | kg in a m3 mixture | Input Variable|
|Fly Ash (component 3) | quantitative | kg in a m3 mixture | Input Variable
|Water (component 4) | quantitative | kg in a m3 mixture | Input Variable|
|Superplasticizer (component 5) | quantitative | kg in a m3 mixture | Input Variable|
|Coarse Aggregate (component 6) | quantitative | kg in a m3 mixture | Input Variable|
|Fine Aggregate (component 7) | quantitative | kg in a m3 mixture | Input Variable|
|Age | quantitative | Day (1~365) | Input Variable|
|Concrete compressive strength | quantitative | MPa | Output Variable|
|&nbsp; &nbsp;|  |  |  |

To summarize, the data set contains;

- 8 input variables (features)
- 1 output variable (labels)
- all the data is numerical
- feature and label names are pretty long
- 1030 instances in the Dataset
- no missing values in the Dataset

There are many versions of this Dataset, and to load it directly, we have selected a [csv version](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv)

## Data Uploading

The first task of this exercise is to upload the data to Azure ML Datastore and make it available for the experiments through the Azure Machine Learning Dataset.here are many ways to create an AzureML Dataset, in this case we will upload a csv file.

Steps for uploading the data to AzureML datastore and registering it as an AzureML Dataframe:

- **Get the workspace**. If a config file was saved, if getting a reference to the workspace can be achieved by calling the **Workspace.from_config** function.
- **Get a reference to the datastore**. The default datastore can be easily retrieved by calling the **workspace.get_default_datastore** function. In this case, we are using a datastore bearing the name of the experiment, and therefore we will use **Datastore.get(workspace, DATASTORE_NAME)**.
- **Upload the data to the datastore**. The concrete strength data is uploaded to our datastore by calling the **datastore.upload** function.
- **Register the data as an AzureML Dataframe** is carried out in two steps by;
  - Create the new dataset by calling the **Dataset.Tabular.from_delimited_files** as the data that we have is tabular. We notice that this function requires a DataPath argument that points to the file that we have just uploaded into the datastore.
  - Finally, **register the dataset** by calling the **Dataset.register** function.

An implementation of the above steps can be found in the following code:

```python{upload_data.py}
'''
Creates concrete dataset to be used as input for the experiment.
'''
import os
from azureml.core import Workspace, Dataset
from azureml.data.datapath import DataPath

# Name of the dataset.
DATASET_NAME = 'concrete_baseline'

# Load the workspace configuration from the .azureml folder.
config_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '.azureml')
w_space = Workspace.from_config(path=config_path)

if DATASET_NAME not in Dataset.get_all(w_space).keys():

    print('Uploading dataset to datastore ...')

    # Getting the default datastore for the workspace.
    data_store = w_space.get_default_datastore()

    # upload all the files in the concrete data folder to the
    # default datastore in the workspace concrete_data_baseline folder
    data_store.upload(
        src_dir=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..',
            '..',
            'data',
            'concrete'),
        target_path='concrete_data_baseline')

    # create a new dataset from the uploaded concrete file
    concrete_dataset = Dataset.Tabular.from_delimited_files(
        DataPath(data_store, 'concrete_data_baseline/concrete.csv'))

    # and register it in the workspace
    concrete_dataset.register(
        w_space,
        'concrete_baseline',
        description='Concrete Strength baseline data (w. header)')

    print('Dataset uploaded')
else:
    print('Dataset already exists')
```

We can verify that the new dataset was correctly created by checking the **Datasets** tab in AzureML Workspace, where we should find the dataset **concrete_baseline**. From this page, we can also see the Datastore where the data was uploaded, which is the default datastore for the workspace. We can also confirm that the data file, **concrete.csv**, was uploaded in the **concrete_data_baseline** folder.

We can also have a look at the dataset by  selecting the **Explore** tab in this page, which directs us to the data dump of the dataset. We can also see that features and label are all numerical, as they all have a **00** near their name. We can also see that the dataset has 1030 instances.

This screen can also give us a quick overview of the data, through the **Preview** tab. Here we can see the distribution, and information like type, minimum and maximum values and the mean and standard deviation for each feature.

If we click on the **datasource** link in the **Details** tab, we are directed to the blob storage where the data is stored. here we can see the **concrete.csv** file.

The figures below show the screens and the links that were used to navigate through the dataset.

![Dataset Upload](/post/img/azureml_pipeline_introduction_dataset_upload.jpg)

## Next Post

In the next post we will 