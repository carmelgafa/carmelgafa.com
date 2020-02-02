---
title: Robotics Toolkit
description: 
date: "2020-02-02T15:13:06+01:00"
jobDate: 2012
work: [robotics, programming, fuzzy-logic]
techs: [C, C-Sharp]
designs: []
thumbnail: robotics_toolkit/robotics_toolkit.png
projectUrl: 
---

The high level design goal of this work is the creation of an autonomous robot navigation toolkit which is flexible, robust, easy to use and to configure and preferably independent of the sensors and actuators utilized. More information about the system can be found in this presentation.

The first component that is required is a physical robot. Although considerable testing and verification of the navigation algorithms used can be achieved through the use of a simulation tool, the only valid validation mechanism is an actual robot with real sensors and actuators.

It is not practical to test every version of the controllers that were developed on real robots, since parameters like battery lifetime or sensor misalignment can affect the results obtained. A simulation tool is therefore a great asset in the development of control algorithms, and was therefore identified as an important component of the system. It is also important to mention that robot sensors have a relevant monetary cost, and consequently their availability is limited when working against a budget. A simulation environment therefore is an excellent tool to construct and optimize hypothetical robots that cannot be easily built in practice due to economic or timescale restrictions. Another advantage of virtual systems over real ones is the inherent lack of restrictions related to environment configuration and sensor limitations. It is very awkward and time consuming to physically build all the obstacles necessary in a large, densely populated environment unless it is constructed virtually. In addition, the physical limitations of sensor mechanisms can be easily altered to prove the relevance of a control concept.

Finally, a standard is necessary so that developers can implement robot controllers that can be easily integrated in this environment. A mechanism must be present so that a library of controllers can be kept, maintained and enhanced. It is desired that the controllers can be deployed on both the simulator and the real robot without any changes.

The following components were developed as parts of this toolkit:

1. Robosim, a generic simulation environment was constructed to verify robot controllers. The simulator can be used to test any control technique that can be used to drive mobile robots.
2. UAN, a mobile robot based on the control methodology described abovewas developed. The implementation necessitated the investigation of interfacing techniques using the I2C protocol, and the implementation of .NET applications onto small embedded systems
3. A number of robot controllers amongst which the implementation of fuzzy behaviour controllers that used context dependent blending to coordinate individual behaviours. A language that can be used to define these controllers was also implemented.
