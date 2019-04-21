import numpy as np
from flask import Flask, request, jsonify
import pickle
import requests
import json

response = requests.get("http://api.worldweatheronline.com/premium/v1/weather.ashx?key=811c892f33bb4688802102448191903&q=mumbai&format=json&num_of_days=5&tp=24")
json_value=response.json()
#print(json_value['data']['weather'][0]['date'])
test=[[0 for i in range(4)]for j in range(5)]
print(test)
for i in range(5):
    for j in range(1):
        humidity=json_value['data']['weather'][i]['hourly'][j]['humidity'];
        precipMM=json_value['data']['weather'][i]['hourly'][j]['precipMM'];
        pressure=json_value['data']['weather'][i]['hourly'][j]['pressure'];
        tempC=json_value['data']['weather'][i]['hourly'][j]['tempC'];
        #windspeedKmph=json.data.weather[i].hourly[j].windspeedKmph;
        test[i][j]=float(humidity);
        test[i][j+1]=float(precipMM);
        test[i][j+2]=float(pressure);
        test[i][j+3]=float(tempC);
        #print(humidity,precipMM,pressure,tempC)
        #print("\n")

print(test)
