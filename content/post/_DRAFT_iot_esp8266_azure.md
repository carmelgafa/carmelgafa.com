---
title: "Iot_esp8266_azure"
date: 2022-03-06
tags: [DRAFT]
draft: true
---


IOT Hub is a platform as a service offering from Microsoft that provides a managed service for bidirectional communication with devices. It integrates very well with other Azure services. In addition, it provides SDKs in several languages such as C#, Python, and C. IOT Hub also supports protocols like MQTT, AMQP, and HTTP.


In Powershell run

```text
az iot hub generate-sas-token --device-id esp8266 --hub-name iothubcg
```

given

```text
{
  "sas": "SharedAccessSignature sr=iothubcg.azure-devices.net%2Fdevices%2Fesp8266&sig=c%2FRc%2FFWGo84UFn8HrPWAywecBYSnke8a%2BLxXZR84lxU%3D&se=1646589211"
}
```




C:\Users\carme>openssl s_client -servername iothubcg.azure-devices.net -connect iothubcg.azure-devices.net:443 | openssl x509 -fingerprint -noout
depth=1 C = US, O = Microsoft Corporation, CN = Microsoft RSA TLS CA 01
verify error:num=20:unable to get local issuer certificate
verify return:1
depth=0 CN = *.azure-devices.net
verify return:1
SHA1 Fingerprint=AF:B9:1B:A1:88:4A:DB:0C:06:97:B0:1D:62:13:0E:9A:49:3E:34:CC