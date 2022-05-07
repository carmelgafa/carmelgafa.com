---
title: "Azureml_pipeline_introduction"
date: 2022-05-05
tags: []
draft: true
description: ""
---

## Introduction

In the previous posts in this series we have examined some of the various features of [Azure Machine Learning](/tags/azure-ml). We have executed some experiments and have seen the results. We will now try to look into a more complete Machine Learning project in Azure, that will utilize the power of other Azure ML features such as:

- **Azure Machine Learning Pipelines**. Azure ML pipelines a way whereby discrete steps (or subtasks) are executed as a workflow. It is important to notice that the concept of pipeline is recurrent in many Azure services. In particular, Microsoft suggest that Azure ML pipelines should be used for Model Orchestration, that a process that produces a model from data. Another different pipeline type in Azure is that used for data processing; where data produces a new data set. In this case, Microsoft suggest products Azure Data Factory or Azure Data Bricks. Nevertheless, it is possible to use Azure ML pipelines to prepare data, if the data transformation is not too complex. Other tasks that can be performed with Azure ML pipelines include:
  - Training configuration
  - Training and validation
  - Repeatable deployments

- **Azure Machine Learning Hyperparameter Tuning**. We will use the HyperDrive package to tune the hyperparameters of our machine learning model.

- **Model Deployment into Azure**. We will deploy our model into Azure as a web service.

## A disclaimer

This article in not intended to be a complete guide to Azure Machine Learning model building and deployment. It is intended to be a starting point for those who are interested in using Azure Machine Learning, and to introduce some of the interesting features available in the platform.

## Structure of these posts

This project has been divided into a number of smaller posts so to limit the length and content of each post. It consists of the following entries:

1. [An introduction to the project, considerations and data uploading to Azure](/post/azureml_pipeline_introduction)
2. [Pipeline Creation](/post/azureml_pipeline_creation)
3. [Hyperparameter Tuning](/post/azureml_pipeline_hyperparameter_tuning)
4. [Model Deployment](/post/azureml_pipeline_model_deployment)

## Folder Structure


## Data Uploading

we will use the UCI Concrete dataset. This dataset contains data on the compressive strength of concrete given the mixture and curing time as features. The dataset is available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength).

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

The Dataset has, therefore:

- 8 input variables (features)
- 1 output variable (labels)
- all the data is numerical
- feature and label names are pretty long
- 1030 instances in the Dataset
- no missing values in the Dataset

There are many versions of this Dataset, and to load it directly, we have selected a [csv version](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv)
