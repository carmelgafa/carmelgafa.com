---
title: "Notes about Azure ML, Part 4 - Creating Azure ML Datasets from a URL"
date: 2022-01-05T11:08:45+01:00
draft: false
tags: [machine-learning, azure ml, dataset, datastore]
---

In a previous post, we discussed how to create a dataset from a datastore, but this is not the only way to create a dataset. This post will examine how to import a dataset from data available on the web.

The process is quite simple, consisting of only three steps:

- Pasting the URL from the provider of the data.
- The data is retrieved and displayed. Changing some of the settings, like the delimiter or if the data contains a header, is possible.
- We can also confirm the type of the fields, and it is possible to remove selected fields from the Azure Ml dataset

The diagram below shows the steps to create a dataset based on the Concrete Strength data from UCI.

![Dataset Creation Process](/post/img/azureml_datasetfromurl_process.jpg)
