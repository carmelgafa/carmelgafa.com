---
title: Robot Uan
description: 
date: "2020-01-09T10:58:22+01:00"
jobDate: 2012
work: [robotics, electronics, programming]
techs: [C, CSharp]
designs: []
thumbnail: uan/robot.jpg
projectUrl: 
tags: ["robot"]
---

{{< figure src="/portfolio/uan/uan_desc.jpg" caption="Robot Uan" >}}

### Design Considerations

The purpose of designing and constructing this robot was to provide a generic test-bed for robot controller design and testing.Unlike most robot platforms used in education or research, UAN does not come with its proprietary development environment but uses Microsoft .NET Mobile edition, thus programming is done using the wide spread .NET framework from Microsoft. 

The challenge addressed by UAN is that changing to a different robot or to a different platform does not necessarily mean a complete change in paradigm, and a substantial part of the knowledge and code can be transferred to the new platform. 

UAN consists of a number of panels connected to a common I2C bus. Power supply to the panels is provided together with the communication cables. The lower level contains the MD22 motor controller, motors, batteries and voltage regulation circuitry. Two 6V 3.2Ah batteries drive the robot; one battery is used to drive the electronic circuits, including the processor . In order to achieve this, its voltage is regulated to 5V using a low drop-out regulator. The other battery is used to drive the motors directly. This approach was preferred over a single battery solution as the voltage level on the processor board is not affected when the motor supply demand increases. 

Three SRF10 sensors are used to detect obstacles in a range in excess of 180 degrees. The SRF10 is very sensitive to objects below the boresight, and the sensors are mounted to aim slightly upwards in order to avoid continuous ground response. The Viper-LITE and Viper-I/O boards are mounted on the topmost panel, since the switches on the Viper-I/O are used to administer the robot, and the CompactFLASH must be easily accessible as it is used to store the robot configuration files.

### Uan Videos
<!-- 
{{< youtube FPV0CT1fllw > caption="Movement check"}}

{{< youtube YXiN4U6D0d0 > caption="Movement test - no sensors"}}

{{< youtube aN-3VG71tAk >  caption="Obstacle Avoidance"}}

{{< youtube B_s7-taF2DY > caption="Wall following"}}

{{< youtube uur0EpSXcQ4 > caption="Complex Behaviour - go to destination and avoid obstacles encountered"}} -->