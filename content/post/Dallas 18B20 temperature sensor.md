---
title: Dallas 18B20 temperature sensor
date: "2020-06-27T16:25:12+02:00"
tags: [arduino]
---

The DS18B20 is a three pin digital thermometer that provides digital temperature measurements as 9 to 12-bit packets over a single wire serial connection. Its temperature range is between -55 and 125 degrees celcius, with very good accuracy of 0.5 degrees between -10 and 85 degrees Celcius.

It can convert a temperature to a 12 bit reading in 750ms.

![ds18b20 pinouts](/note/arduino/ds18b20_pinouts.jpeg)


The device datasheet can be downloaded from [here](https://datasheets.maximintegrated.com/en/ds/DS18B20.pdf).

### Required libraries to use the DS18B20

In order to use the temperature sensor, on e muct include the [Arduino Library for Maxim Temperature Integrated Circuits](https://github.com/milesburton/Arduino-Temperature-Control-Library) developed by Miles Burton. Constructing a DallasTemperature instance requires two a reference to a [OneWire](https://playground.arduino.cc/Learning/OneWire/) object. OneWire is another library developed by MIles Burton, that provides onewire or MicroLan capabilities. Costructing a OneWire object simply requires the ardiuno pin that will communicate to the single wire device.

The program to read the temperature will therefore look as follows:

``` C
#include <OneWire.h>
#include <DallasTemperature.h>

// use pin 10 to read temperature
OneWire oneWire(10);

DallasTemperature dallasTemperature(&oneWire);


void setup()
{
  Serial.begin(9600);
  sensor.begin();
}

void loop()
{
  dallasTemperature.requestTemperatures();

  // as we only have one device, the index of the thermometer
  //is 0
  float temp = dallasTemperature.getTempCByIndex(0);

  Serial.print("Temperature: ");
  Serial.println (temp);

  delay(800);
}
```
