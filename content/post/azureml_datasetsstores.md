---
title: "Notes about Azure ML - Datasets and Datastores"
date: 2021-12-23
tags: [machine-learning, azureml, dataset, datastore]
draft: true
---

In AzureML, the two important concepts that help us work with data are Datastores and Datasets.

- Datastores are used for storing connection information to Azure storage services
- Datasets are references to the location of the data source.

Azure has various locations for storing data. such as;

- Azure Blob Storage
- Azure SQL Database
- Azure Datafactory
- Azure Databricks

These are the places where the data can exist.

An Azure storage account is a container for all the Azure Storage data objects blobs, file shares, queues, tables, and disks, making them accessible from anywhere in the world over HTTP or HTTPS.

When we create a an AzureML resource, an associated storage account is created as well. This account will contain two built-in datastores;

- an Azure Storage Blob Container and
- an Azure Storage File Container

which will contain the relevant data and code for the AzureML resource.
