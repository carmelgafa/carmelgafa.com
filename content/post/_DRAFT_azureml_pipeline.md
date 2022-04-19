---
title: "DRAFT_azureml_pipeline"
date: 2022-04-11
tags: []
draft: true
description: ""
---

In the last post of this series, we trained a simple model to predict the compressive strength of concrete. The model what somewhat trivial and also quite unstructured as we ended up using scikit-learn pipelines to train the model. Azure ML does provide a way to structure ML projects in distinct computational tasks that when connected together form a pipeline. Pipelines can be created in one of two ways:

- Using the designed tool in Azure ML. This offers a drag and drop interface to create pipelines in a no-code environment.
- Using the Azure ML SDK. This offers a more structured interface to create pipelines in a code environment.

Pipelines offer a number of benefits over azureml scripts discussed in the previous post, namely:

- Team efficiency. Pipeline construction can be done in a team environment, where the pipeline can be shared and used by multiple teams.




Key advantage	Description
Unattended runs	Schedule steps to run in parallel or in sequence in a reliable and unattended manner. Data preparation and modeling can last days or weeks, and pipelines allow you to focus on other tasks while the process is running.
Heterogenous compute	Use multiple pipelines that are reliably coordinated across heterogeneous and scalable compute resources and storage locations. Make efficient use of available compute resources by running individual pipeline steps on different compute targets, such as HDInsight, GPU Data Science VMs, and Databricks.
Reusability	Create pipeline templates for specific scenarios, such as retraining and batch-scoring. Trigger published pipelines from external systems via simple REST calls.
Tracking and versioning	Instead of manually tracking data and result paths as you iterate, use the pipelines SDK to explicitly name and version your data sources, inputs, and outputs. You can also manage scripts and data separately for increased productivity.
Modularity	Separating areas of concerns and isolating changes allows software to evolve at a faster rate with higher quality.
Collaboration	Pipelines allow data scientists to collaborate across all areas of the machine learning design process, while being able to concurrently work on pipeline steps.


In this post, we will go through a machine learning exercise through the use of Azure ML pipelines. It is important to immediately claim that the objective of this post is not to create the best ML pipeline or the best model for this example, but rather to attempt to go through the features of Azure ML pipelines and how they can be used is such context.

We will use the [Concrete Strength Prediction](https://www.kaggle.com/c/concrete-compressive-strength) dataset from Kaggle. The dataset contains concrete compressive strength measurements from a variety of concrete samples. The goal is to predict the compressive strength of concrete samples. Again, there is no particular reason why this particular dataset was used, except from the fact that it is small and simple enough to be easily worked with.

### Plan of Action

We will start this discussion by implementing a simple training pipeline that will train a number of models and selects the best one. The selected model is then picked up by a second pipeline that fine-tunes the model and then registers it so that it can be used in production. The model is then deployed as an Azure ML endpoint and consumed.

### Passing Data through Pipeline Steps

The process of chaining steps togetehr to form a pipeline obviously also entails data that is manipulated in each step as passed on to the subsequent step. **OutputFileDatasetConfig** can be used for this purpose but also as a mechanism to persist data for later use. It is capable of writing data into fileshare, blob storage and also data-lake. It supports two modes of operation:

- **mount**, where files are written and permanently stored in the target directory.
- **upload**, where files are written and permanently stored in the target directory at the end of the job; which may mean that the files are not available if the job fails.

We define **OutputFileDatasetConfig** objects, as part of the pipeline definition. If, as an example we need a place to store train and test data that is used by the various steps;

```python
from azureml.data.output_dataset_config import OutputFileDatasetConfig

train_folder = OutputFileDatasetConfig('train_folder')
test_folder = OutputFileDatasetConfig('test_folder')
```

The **OutputFileDatasetConfig** objects are the passed as arguments to the **PythonScriptStep** that require them

```python
step_train_test_split = PythonScriptStep(
    script_name = 'train_test_split.py',                            # name of the script to run
    arguments=[                                                     # arguments to the script
        '--train_folder', train_folder,
        '--test_folder', test_folder],
    inputs = [<<Dataset containing baseline data>>],
    compute_target = <<compute target name>>,
    source_directory=<<directory where the script is located>>, 
    allow_reuse = True,                                            # reuse the step if possible
    runconfig = run_config                                         # run configuration
)
```
