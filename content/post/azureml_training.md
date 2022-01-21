---
title: "Azureml_training"
date: 2022-01-19
tags: []
draft: true
---

In a [previous post]() we have seen how to create a ml workspace and provision a compute resource in Azure using AzureML SDK.

In the post we will see how to execute training and deployment of a model in the ML environment using AzureML SDK.

We will stat by executing a very simple experiment that will just print a message. The steps required to execute this trivial experiment are:

- Create and **Experiment** instance. This requires a reference to the workspace and the name of the experiment.

- Create a configuration information **ScriptRunConfig** instance that packages the configuration information necessary to execute the experiment. Such information  includes 
    - the script to execute
    - the arguments to the script
    - the environment to run the script on

- submit the experiment, specifying the configuration information.

- run the experiment

Therefore the script necessary to execute our trivial experiment on a compute target is as follows:

```python

from azureml.core import Workspace, Experiment, ScriptRunConfig
import os
import constants

def run_experiment():

    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.azureml')
    ws = Workspace.from_config(path=config_path)
    
    
    experiment = Experiment(ws, constants.EXPERIMENT_NAME)
    
    config = ScriptRunConfig(
        source_directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src_1'),
        script='experiment_1.py',
        compute_target=constants.TARGET_NAME)

        
    run = experiment.submit(config)
    aml_run = run.get_portal_url()
    
    print(aml_run)
    
    
if __name__ == '__main__':
    run_experiment()
```

It is important to note that:

- config.json file necessary to get a reference to the workspace is located into the _.azureml_ folder.
- the script containing the code to execute is located into the _src_1_ folder.
- the script name is _experiment_1.py_.

The **submit** method of the **Experiment** class returns a Run instance. This instance contains the information necessary to access the results of the experiment, including the URL of the portal to access the results.

Upon execution, the script is put in a docker container and executed on the compute target. The output of the script can be read in the experiment log.