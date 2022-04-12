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