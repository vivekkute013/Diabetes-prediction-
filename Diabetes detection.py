# -*- coding: utf-8 -*-
"""
@ Module to detect Diabetes using Decision Tree Algorithm

Created on Wed Jul 20 20:00:27 2022

@author: Vivek Kute
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("D:\Machine Learning\dataset_files/diabetes.csv")
X = data.iloc[:, : -1].values
Y = data.iloc[:, 8].values
Y = np.reshape(Y, (768, 1))

count = data['Outcome'].value_counts()

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state = 0)
reg.fit(X, Y)

# Prediction
Prediction = reg.predict([[9,171,110,24,240,45.4,0.721,54]])
if(Prediction[0] == 0):
    print("Patient has no diabetes")
else:
    print("Patient has Diabetes")
    
    
#Evalution of predictions
....
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
