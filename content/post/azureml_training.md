---
title: "Training Models in Azure ML Part 6 - Experiment Creation"
date: 2022-03-13
tags: [machine-learning, azure ml, experiment]
draft: false
---

In a [previous post](/post/azureml_sdk_workspace/), we have seen how to create an ML workspace and provision a compute resource in Azure using AzureML SDK; now we will see how to execute an experiment the ML environment using AzureML SDK.

We will start by executing a simple experiment that will print a message. The steps required to run this trivial experiment are:

- Create an experiment script to be executed.

- Create and **Experiment** instance. This requires a reference to the workspace and the name of the experiment.

- Create a configuration information **ScriptRunConfig** instance that packages the configuration information necessary to execute the experiment. Such information  includes:
  - the script to execute
  - the arguments to the script
  - the environment to run the script on

- submit the experiment, specifying the configuration information.

- run the experiment

### The Experiment

An essential task in this process is to create the script to execute the experiment. In our case, the following procedure is observed for all experiments:

- A folder having the name of the experiment is created, for example, **Experiment_1**.
- This folder will contain all the scripts related to the experiment, particularly the script to execute the experiment. The script to execute the experiment is called **experiment_1.py**.

In our simple example, the script to execute the experiment is as follows:

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

    config = ScriptRunConfig(
        source_directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Experiment_1'),
        script='experiment_1.py',
        compute_target=constants.TARGET_NAME)

    experiment = Experiment(ws, constants.EXPERIMENT_NAME)
    run = experiment.submit(config)

    aml_run = run.get_portal_url()
    print(aml_run)

if __name__ == '__main__':
    run_experiment()
```

It is important to note that:

- config.json file necessary to get a reference to the workspace is located in the **.azureml** folder.
- the script containing the experiment to execute is located in the **Experiment_1** folder as explained before and the script name is **experiment_1.py**.

The **submit** method of the **Experiment** class returns a Run instance. This instance contains the information necessary to access the experiment results, including the URL of the portal to access the results.

Upon execution, the script is put in a docker container and executed on the compute target. We can read the output of the script in the experiment log.

### Environment

The example above is elementary and of little use, but it shows the basic steps to execute a script on a compute target. To run more useful experiments, we will need to create a more complex environment that will include the libraries and the code necessary to execute the experiment. Environments are stored and tracked in your AzureML workspace, and upon creation, the workspace will already contain typical environments normally used in ML projects.

We can also create environments specific to our project through an **Environment** instance. Two interesting methods in the **Environment** class are:

- **from_conda_specification**. This method allows creating an environment from a conda specification stored in a YAML file. A typical specification is as follows:

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

- **from_pip_specification**. This method allows creating an environment from a pip specification stored in a text file. A typical specification, in this case, is as follows:

``` txt
python==3.8.10
torch==1.10.1
torchvision==0.11.2
numpy==1.19.4
```

To execute an experiment that requires our environment, we must provide an **Environment** instance to the **ScriptRunConfig** instance. This is done as follows:

```python
from importlib.resources import path
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
import os
import constants

def run_experiment():

    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.azureml')
    ws = Workspace.from_config(path=config_path)

    config = ScriptRunConfig(
        source_directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Experiment_1'),
        script='experiment_1.py',
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

    experiment = Experiment(ws, constants.EXPERIMENT_NAME)
    run = experiment.submit(config)

    aml_run = run.get_portal_url()
    print(aml_run)

if __name__ == '__main__':
    run_experiment()
```

Note that the code above is similar to the previous example, but we have now added the **Environment** instance to the **ScriptRunConfig** instance.

### Execution

Once executed, the script will return a Run instance. This instance contains the information necessary to access the experiment results, including the URL of the portal to access the results.

The URL directs the user to the experiment portal, where we can access the experiment results. This screen links to the environment that we specified, including the status of the experiment run and the name of the script being executed.

![Experiment Portal](/post/img/azureml_training_execution1.jpg)

Other tabs provide additional information about the experiment run. In the snapshot tab, we can see the script that was executed.

![Experiment Portal](/post/img/azureml_training_execution2.jpg)

Logs are also available in the portal, in the **Outputs+Logs** tab. These provide helpful information about the execution of the experiment, especially when the experiment fails. We can also see the output of our experiment script.

![Experiment Portal](/post/img/azureml_training_execution3.jpg)

Going one level up, we can see information about the various execution runs of the experiment, their status, compute target, and the time taken to execute them.

![Experiment Portal](/post/img/azureml_training_execution4.jpg)

### Conclusion

In this post, we have seen the steps required to execute an ML experiment in AzureML. The steps required are:

- Create a folder to contain the experiment.
- Create a script to execute the experiment.
- Create a script to execute the experiment.
  - Create a requirements file containing the libraries and the code necessary to execute the experiment. The file can be either:
    - a pip specification (txt file).
    - a conda specification (YAML file).
  - Create a **ScriptRunConfig** instance that packages the configuration information necessary to execute the experiment. The configuration information includes:
    - the folder containing the script to execute.
    - the script to execute.
    - the Azure ML compute to execute the experiment upon.
  - Create an **Environment** instance that packages the libraries and the code necessary to execute the experiment. The environment will require the file that we created in the previous step.
  - Set the **Environment** instance in the **ScriptRunConfig** instance using **run_config.environment**.
  - Create an **Experiment** instance, giving it a name and a reference to the workspace.
  - Submit the experiment using the **ScriptRunConfig** instance as the parameter. The **submit** method returns a **Run** instance.
