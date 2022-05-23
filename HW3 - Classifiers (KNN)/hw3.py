#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:50:52 2021

@author: chrisnguyen
"""

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

## Split features from label 
heart_columns = heart_dt.columns
heart_dt_features = heart_dt[heart_columns[:11]]
heart_dt_label = heart_dt[heart_columns[11]]

## Standardize features

def std (dataframe):
    dataframe_std = np.copy(dataframe)
    for i in range(len(dataframe.columns)):
        dataframe_std[:, i] = (dataframe.iloc[:, i] - dataframe.iloc[:, i].mean()) / dataframe.iloc[:, i].std()
    dataframe_std = pd.DataFrame(dataframe_std, columns= dataframe.columns)
    return dataframe_std

heart_dt_features_std = std(heart_dt_features)

## Split data into training, development, test

def split_data (data_frame_features,data_frame_labels,a,b,c):
    # takes the data frame, randomly shuffles the data, splits the data from 0 to a then a to b then b to end.
    if a + b + c != 1:
        return 
    indexes = [int(a*len(data_frame_features)), int((a+b)*len(data_frame_features))]
    return np.split(data_frame_features.sample(frac=1, random_state=42),indexes) + np.split(data_frame_labels.sample(frac=1, random_state=42),indexes)

heart_training_X, heart_dev_X, heart_test_X, heart_training_y, heart_dev_y, heart_test_y  = split_data(heart_dt_features_std,heart_dt_label,0.7,0.15,0.15)

## fuction for predicted stats

def accuracy(predicted_label,true_label):
    error = 0
    right = 0
    for i in range(len(true_label)):
        if predicted_label[i] == true_label[i]:
            right += 1
    percent_right = right/len(predicted_label) *100
    return (right,percent_right)

def stats(predicted_label,true_label):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(true_label)):
        if predicted_label[i] == true_label[i] == 1:
            tp += 1 
        elif predicted_label[i] == true_label[i] == 0:
            tn += 1
        elif predicted_label[i] != true_label[i] and predicted_label[i] == 1:
            fp += 1    
        elif predicted_label[i] != true_label[i] and predicted_label[i] == 0:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall)/(precision + recall)
    return (precision, recall, f1)

## Part 1 - logistic regression with default classifier parameters
from sklearn.linear_model import LogisticRegression

part1 = LogisticRegression(random_state=0).fit(heart_training_X, heart_training_y)

print('Part 1 - default model')
print(accuracy(part1.predict(heart_dev_X),heart_dev_y.tolist()))
print(stats(part1.predict(heart_dev_X),heart_dev_y.tolist()))

## Part 2 - tweak the classifier parameters 
most_accurate = (0,0)
# [10000,1000,100,10,1,0.1,0.01,0.001,0.0001,0.00001]
for i in [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,5,10,100,1000,10000]:    
    part2 = LogisticRegression(random_state=0,C=i).fit(heart_training_X, heart_training_y)
    accuracy_part2 = accuracy(part2.predict(heart_dev_X),heart_dev_y.tolist())

    if accuracy_part2[1] > most_accurate[1]:
        most_accurate = (i,accuracy_part2[1])
part2 = LogisticRegression(random_state=0,C=most_accurate[0]).fit(heart_training_X, heart_training_y)


print('\nPart 2 - tweaked model')
print(accuracy(part2.predict(heart_dev_X),heart_dev_y.tolist()))
print('C:' + str(most_accurate[0]))

## Part 3 - KNN
class KNN(object):
    
    def __init__(self):
        self.tranining_X = []
        self.tranining_y = []

    def fit(self,X,y):
        self.tranining_X = X
        self.tranining_y = y
        return self
    
    def predict (self,x,k):
        y_predicted = []
        temp_df = self.tranining_X.copy(deep=True)
        x = x.reset_index(drop=True)
        temp_df = temp_df.reset_index(drop=True)
        for i in range(len(x)):
            distance = [0]*len(temp_df.index)
            for col in temp_df.columns:
                distance += (temp_df[col] - [x.iloc[i][col]]*len(temp_df.index))**2 
            distance_index = np.sqrt(distance).sort_values(axis=0)[:k].index
            if self.tranining_y.iloc[distance_index].to_list().count(-1) > self.tranining_y.iloc[distance_index].to_list().count(1):
                y_predicted.append(-1)
            else :
                y_predicted.append(1)            
        return y_predicted

part3 = KNN()
part3.fit(heart_training_X, heart_training_y)   
most_accurate_part3 = (0,0)
for i in [1,3,5,7,9,11,13,15]:    
    accuracy_part3 = accuracy(part3.predict(heart_dev_X,i),heart_dev_y.tolist())

    if accuracy_part3[1] > most_accurate_part3[1]:
        most_accurate_part3 = (i,accuracy_part3[1])

print('\nPart 3 - KNN')
print(accuracy(part3.predict(heart_dev_X,most_accurate_part3[0]),heart_dev_y.reset_index(drop=True)))
print('k:' + str(most_accurate_part3[0]))


## Part 4 - baseline classifier
from sklearn.dummy import DummyClassifier

part4_mf = DummyClassifier(strategy="most_frequent").fit(heart_training_X, heart_training_y)
print('\nPart 4 - baseline classifiers')
print(accuracy(part4_mf.predict(heart_dev_X),heart_dev_y.tolist()))

part4_st = DummyClassifier(strategy="stratified").fit(heart_training_X, heart_training_y)
print(accuracy(part4_st.predict(heart_dev_X),heart_dev_y.tolist()))

## Part 5 - Compare best model 
print('\nPart 5 \nAccuracy')
print('default logistic: ' + str(accuracy(part1.predict(heart_test_X),heart_test_y.tolist())))
print('tweaked logistic: ' + str(accuracy(part2.predict(heart_test_X),heart_test_y.tolist())))
print('knn: ' + str(accuracy(part3.predict(heart_test_X,most_accurate_part3[0]),heart_test_y.tolist())))
print('most_frequent: ' + str(accuracy(part4_mf.predict(heart_test_X),heart_test_y.tolist())))
print('stratified: ' + str(accuracy(part4_st.predict(heart_test_X),heart_test_y.tolist())))

print('\nF1')
print(stats(part1.predict(heart_test_X),heart_test_y.tolist()))
print(stats(part2.predict(heart_test_X),heart_test_y.tolist()))
print(stats(part3.predict(heart_test_X,most_accurate_part3[0]),heart_test_y.tolist()))
print(stats(part4_mf.predict(heart_test_X),heart_test_y.tolist()))
print(stats(part4_st.predict(heart_test_X),heart_test_y.tolist()))
