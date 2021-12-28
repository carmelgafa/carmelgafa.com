---
title: "Notes about Azure ML, Part 3 - Azure Machine Learning SDK, working with Workspaces, Computes, Datasets and Datastores"
date: "2021-12-28T16:50:00+01:00"
tags: [machine-learning, azure ml, dataset, datastore]
draft: true
---

The Azure Machine Learning SDK for Python enables us to interact with the Azure Machine Learning service using a  Python environment. In this post we will discuss how to create, manage and use Azure Machine Learning Workspaces, Computes, Datasets and Datastores using the Azure Machine Learning SDK for Python.

#### Create a Workspace

Creating a workspace is shown below. The **create** method required a name for the workspace, a subscription ID, and a resource group (that can be created for the workspace by setting the **create_resource_group** flag to true), and a location.

To use the workspace it is useful to create a JSON file with the workspace details so that it can be easily reused. This is done through the **write_config** method. The workspace details are written to a file called **config.json** and this can be used to get a workspace object by using the **Workspace.from_config** method.

```python
from azureml.core import Workspace
import os

SUBSCRIPTION_ID = 'copy and paste subscription id here'

config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.azureml')

ws = Workspace.create(name = 'wsdeleteme',
                    subscription_id = SUBSCRIPTION_ID,
                    resource_group = 'rgdeleteme',
                    create_resource_group=True,
                    location = 'West Europe')

ws.write_config(path=config_path, file_name="config.json")
```

#### Create Compute Instance

Creating a compute instance is shown below. It starts with an attempt to use the instance through the **ComputeInstance** constructor, that requires a workspace object and a name for the instance. The Contructor throws an exception if the instance does not exist, and in this care the instance is created. This is achieved by first creating a provisioning configuration object, which is then used to create the instance by using **ComputeInstance.provisioning_configuration**. The provisioning configuration object is then used to create the instance by using **ComputeInstance.create**. We wait for the instance to be created by using **ComputeInstance.wait_for_completion**.

```python
from azureml.core import Workspace
import os
from azureml.core.compute import ComputeInstance
from azureml.core.compute_target import ComputeTargetException

config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.azureml')

ws = Workspace.from_config(path=config_path)

instance_name = 'instdeleteme'
vm_size='Standard_DS11_v2'

try:
    compute_instance = ComputeInstance(workspace=ws, name=instance_name)
    print("Found existing compute instance. Using it...")
except ComputeTargetException:
    compute_config = ComputeInstance.provisioning_configuration(vm_size=vm_size,
                                            ssh_public_access=False)

    compute_instance = ComputeInstance.create(ws, instance_name, compute_config)

    compute_instance.wait_for_completion(show_output=True)
```

#### Create Compute Target

Compute targets are created in a similar way to compute instances, but in this case an Azure ML compute manager, **AmlCompute** is used to create the compute target. Notice that in this case the target was creating in the low priority tier, and our quota allows us to have a maximum of 2 nodes od the the selected vm size.

```python
from azureml.core import Workspace
import os
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.azureml')

ws = Workspace.from_config(path=config_path)

target_name = 'targdeleteme'
vm_size='Standard_DS11_v2'


try:
    compute_target = ComputeTarget(workspace=ws, name=target_name)
    print("Found existing compute target. Using it...")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                            vm_priority='lowpriority',
                                            idle_seconds_before_scaledown=120,
                                            min_nodes=0,
                                            max_nodes=2)
    compute_target = ComputeTarget.create(ws, target_name, compute_config)
    
    compute_target.wait_for_completion(show_output=True)
```
