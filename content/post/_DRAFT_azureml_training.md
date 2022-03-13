---
title: "Training Models in Azure ML Part 6 - Experiment Creation"
date: 2022-03-13
tags: [DRAFT]
draft: true
---

In a [previous post](/post/azureml_sdk_workspace/) we have seen how to create a ml workspace and provision a compute resource in Azure using AzureML SDK; now we will see how to execute an experiment the ML environment using AzureML SDK.

We will stat by executing a very simple experiment that will just print a message. The steps required to execute this trivial experiment are:

- Create an experiment script to be executed.

- Create and **Experiment** instance. This requires a reference to the workspace and the name of the experiment.

- Create a configuration information **ScriptRunConfig** instance that packages the configuration information necessary to execute the experiment. Such information  includes 
  - the script to execute
  - the arguments to the script
  - the environment to run the script on

- submit the experiment, specifying the configuration information.

- run the experiment

### The Experiment

An important task in this process is create the script to actually execute the experiment. In our case the following procedure is observed for all experiments:

- A folder having the name of the experiment is created, for example **Experiment_1**.
- This folder will contain all the scripts related to the experiment, in particular the script to execute the experiment. The script to execute the experiment is called **experiment_1.py**.

In our simple example the script to execute the experiment is as follows:

```python
print('Experiment Executed!')
```

### Script Execution

The script to execute our trivial experiment on a compute target is created as follows:

```python

from azureml.core import Workspace, Experiment, ScriptRunConfig
import os
import constants

def run_experiment():

    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.azureml')
    ws = Workspace.from_config(path=config_path)

    experiment = Experiment(ws, constants.EXPERIMENT_NAME)
    
    config = ScriptRunConfig(
        source_directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Experiment_1'),
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
- the script containing the experiment to execute is located into the **Experiment_1** folder as explained before and  the script name is **experiment_1.py**.

The **submit** method of the **Experiment** class returns a Run instance. This instance contains the information necessary to access the results of the experiment, including the URL of the portal to access the results.

Upon execution, the script is put in a docker container and executed on the compute target. The output of the script can be read in the experiment log.

### Environment

The example above is very simple and of little use but it shows the basic steps to execute a script on a compute target. In order to run more useful experiments, we will need to create a more complex environment, that will include the libraries and the code necessary to execute the experiment. Environments are stored and tracked in your AzureML workspace; and upon creation the workspace will already contain typical environments that are normally used in ML projects.

We can also create  our own environments that are specific to our project, through an **Environment** instance. Two interesting methods in the **Environment** class are:

- **from_conda_specification**. This method allows to create an environment from a conda specification that is stored in a yaml file. A typical specification is as follows:

``` yaml
name: experiment-env
channels:
  - defaults
  - pytorch
dependencies:
  - python=3.8.10
  - pytorch=1.10.1
  - torchvision=0.11.2
  - numpy=1.19.4
```

- **from_pip_specification**. This method allows to create an environment from a pip specification stored in a txt file. A typical specification in this case is as follows:

``` txt
python==3.8.10
torch==1.10.1
torchvision==0.11.2
numpy==1.19.4
```

In order to execute an experiment that requires our environment, we will now need to provide an **Environment** instance to the **ScriptRunConfig** instance. This is done as follows:

```python
from importlib.resources import path
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
import os
import constants

def run_experiment():

    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.azureml')
    ws = Workspace.from_config(path=config_path)

    experiment = Experiment(ws, constants.EXPERIMENT_NAME)

    config = ScriptRunConfig(
        source_directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src_2'),
        script='experiment_2.py',
        compute_target=constants.TARGET_NAME)

    # env = Environment.from_conda_specification(
    #     name = 'env-2',
    #     file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_experiment_2.yml')
    # )
    
    env = Environment.from_pip_requirements(
        name = 'env-2',
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')
    )

    config.run_config.environment = env
    run = experiment.submit(config)
    aml_run = run.get_portal_url()
    print(aml_run)

if __name__ == '__main__':
    run_experiment()
```

Note that the code above is very similar to the previous example, but we have now added the environment to the ScriptRunConfig instance.

