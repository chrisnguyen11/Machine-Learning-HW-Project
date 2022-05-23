#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:22:16 2021

@author: chrisnguyen
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# https://www.kaggle.com/fedesoriano/heart-failure-prediction?select=heart.csv
heart_dt = pd.read_csv('heart.csv')

## making features numerical 
heart_dt['Sex'] = heart_dt['Sex'].map({'F':-1,'M':1})
heart_dt['ChestPainType'] = heart_dt['ChestPainType'].apply(lambda x: 0 if x == 'ASY' else (1 if x == 'NAP' else (2 if x == 'ATA' else 3)))
heart_dt['RestingECG'] = heart_dt['RestingECG'].apply(lambda x: 0 if x == 'Normal' else (1 if x == 'ST' else 2))
heart_dt['ExerciseAngina'] = heart_dt['ExerciseAngina'].map({'Y':0,'N':1})
heart_dt['ST_Slope'] = heart_dt['ST_Slope'].apply(lambda x: 0 if x == 'Up' else (1 if x == 'Flat' else 2))
heart_dt['HeartDisease'] = heart_dt['HeartDisease'].map({0:-1,1:1})

heart_columns = heart_dt.columns
heart_dt_features = heart_dt[heart_columns[:11]]
heart_dt_label = heart_dt[heart_columns[11]]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(heart_dt_features),columns=heart_columns[:11])

## Part 1 - Split Data: training and test
from sklearn.model_selection import train_test_split
heart_training_X, heart_test_X, heart_training_y, heart_test_y= train_test_split(scaled_data, heart_dt_label, test_size=0.20, random_state=42)

## Part 2 - Classifier and n-fold cross validation    
from sklearn.linear_model import LogisticRegression

def accuracy(predicted_label,true_label):
    error = 0
    right = 0
    for i in range(len(true_label)):
        if predicted_label[i] == true_label[i]:
            right += 1
    percent_right = right/len(predicted_label) *100
    return (right,percent_right)

def nfold_cross_valid(X, y, k, c,p):
    X_fold , y_fold = np.array_split(X,k), np.array_split(y,k)
    evaluation = []
    for K in range(k):
        X_test , y_test = X_fold[k - K -1], y_fold[k - K -1]
        X_train , y_train = pd.concat(X_fold[:k - K -1] + X_fold[k - K :]), pd.concat(y_fold[:k - K -1] + y_fold[k - K :])
        k_model = LogisticRegression(random_state=0,solver='saga',penalty='elasticnet',C=c,l1_ratio=(p)).fit(X_train, y_train)
        evaluation.append((accuracy(k_model.predict(X_test),y_test.tolist())[1]))
    return np.mean(evaluation)

print(nfold_cross_valid(heart_training_X,heart_training_y,5,1,1))

## Part 3 - Grid Search 
def grid_search(X,y,k):
    c_values = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
    p_values = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    evaluation = []
    for c in c_values:
        for p in p_values:
            evaluation.append((c,p,nfold_cross_valid(X, y, k, c, p)))
    return pd.DataFrame(evaluation)

maxtrix_grid = grid_search(heart_training_X,heart_training_y,5)
most_accurate = maxtrix_grid.iloc[maxtrix_grid[2].idxmax()]
print(maxtrix_grid.iloc[maxtrix_grid[2].idxmax()])

## Part 4 - Model Evaluation 
part4 = LogisticRegression(random_state=0,solver='saga',penalty='elasticnet',C=most_accurate[0],l1_ratio=(most_accurate[1])).fit(heart_training_X, heart_training_y)
print(accuracy(part4.predict(heart_test_X),heart_test_y.tolist())[1])
