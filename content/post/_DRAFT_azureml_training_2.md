---
title: "Training Models in Azure ML Part 7 - The Concrete Strength Dataset"
date: 2022-03-15
tags: [machine-learning, azure ml, experiment]
draft: true
---

In the [previous post](/post/azureml_training/) of this series, we have seen how to create an execute a machine learning experiment in AzureML. Our experiment was by no means a real ML experiment, but a simple script that printed a message.

In this post, we will see how to create and execute a ML experiment that actually involves training a model. It is important to note that this post is just the first step in the process of creating a proper ML pipeline in AzureML and is therefore not the best of solutions. In future posts, we will continue improving this process and create a better ML pipeline in AzureML.

## Situation

We will consider the [Concrete Compressive Strength Data Set](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength) from UCI Machine Learning Repository. The data set contains the following features:

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

The Dataset has therefore:

- 8 input variables (features)
- 1 output variable (labels)
- all the data is numerical
- feature and label names are quite long
- 1030 instances in the dataset
- no missing values in the dataset

There are many versions of this dataset and in order to load it directv we have selected a [csv version](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv)

## Loading the data into an AzureML dataset

