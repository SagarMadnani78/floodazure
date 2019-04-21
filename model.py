import pandas as pd  
import numpy as np
import requests
import json
import pickle


names = ['date','maxtemp','mintemp','time','humidity','precipMM','pressure','tempc','windspeedKmph','cloudcover','classLabel']
dataset = pd.read_csv("C:/Users/SAHIL/Desktop/FloodPredict/Updated1.2 Mumbai.csv",names=names)  
dataset.head() 

#from imblearn.over_sampling import SMOTE

#sm = SMOTE(random_state=42)
#X_balanced, y_balanced = sm.fit_sample(X, y)
from sklearn.utils import resample
dataset_majority=dataset[dataset.classLabel==0]
dataset_minority=dataset[dataset.classLabel==1]

dataset_min_up=resample(dataset_minority,replace=True,n_samples=10488, random_state=123)
dataset_up=pd.concat([dataset_majority,dataset_min_up])

#print(dataset_up.classLabel.value_counts())
X = dataset.iloc[:, [4,5,6,7]].values  
y = dataset.iloc[:, 10].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
#print(y_test.shape)
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[1.8]]))
"""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cnf_matrix=confusion_matrix(y_test,y_pred.round())
print(cnf_matrix)  
print(classification_report(y_test,y_pred.round()))  
print(accuracy_score(y_test, y_pred.round()))
"""


