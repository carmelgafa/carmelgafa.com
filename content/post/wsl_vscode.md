---
title: "Developing in the WSL with VSCode"
date: 2021-10-10
tags: ["development", "wsl", vscode]
draft: false
---

The Windows Subsystem for Linux lets developers run a Linux environment on  a Windows OS. This is considerably lighter than VMs. This is the first post of a series that will detail the deployment of python applications in a linux container on Azure.

- Installation of WSL can be carried out by following the steps listed [here](https://docs.microsoft.com/en-gb/windows/wsl/install-win10#step-4---download-the-linux-kernel-update-package).

- It is necessary to enable two features so that the WSL environment can be used:
  - VM Platform
  - Windows subsystem for Linux.

  They can be verified (or enabled) from the Windows features dialog (type optionalfeatures.exe in CMD prompt)

    ![optionalfeatures](/post/img/wsl_image1.png)

  - Set WSL 2 as default version - performance is much better.
  - A Linux 2distribution is required to be installed on Window. In this example I will refer to an Ubuntu distro. These can be downloaded for Widows store or directly from [here](https://docs.microsoft.com/en-us/windows/wsl/install-manual). After installation a username and password for the Linux system are requested.

- Install Docker desktop from [here](https://docs.docker.com/docker-for-windows/install/) so that there is good visibility of Docker containers.

I use VSCode as my preferred IDE, and a couple of extensions are necessary on this platform:

- Obviously the [Docker Extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker), so that docker containers are properly managed.
- [Remote - WSL Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) so that folder manipulation can be carried out on the Windows subsystem for Linux.

Connecting to the Windows Subsystem for Linux from VSCode is then very easy

- Click on the Open a Remote Window button

![connectvscode](/post/img/wsl_image2.png)

- Select **New WSL Window command** or **New WSL using Distro...** if multiple Linux distributions are installed. This will open a new VSCode window running connected to WSL.

Upon selecting a project folder and  and opening (or creating) a python file, VSCode will ask to install the recommended extensions for python in this subsystem. Once the Python extensions are installed, the Python interpreter will be selected.

Being a 'new' subsystem, python libraries must be installed. The following steps should be executed if the **pip** command will not work:

- Update package info; `sudo apt-get update`
- Download all upgrades; `sudo apt-get upgrade`
- Reinstall pip; `sudo apt-get install python3-pip`

This should enable commands like `pip install numpy` to work.

As a test, we use the following simple program that adds two numpy arrays.

```python
import numpy as  np

a=np.random.rand(3,3)
b=np.random.rand(3,3)
 
print(a)
print(b)
print(a+b)
```

The application can be executed by using the usual command **Python: Run Python file in terminal** or by keying in **python3 <<filename.py>>** in the terminal.
