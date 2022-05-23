#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:45:02 2021

@author: chrisnguyen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 13:58:54 2021

@author: chrisnguyen
"""
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(1)

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk

protein_dt = pd.read_csv('herg_features.tsv',sep='\t',skiprows=(1))

protein_columns = protein_dt.columns
protein_dt_features = protein_dt[protein_columns[9:11]]
protein_dt_label = protein_dt[protein_columns[12]]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(protein_dt_features),columns=protein_columns[9:11])


def split_data (data_frame_features,data_frame_labels,a,b,c):
    # takes the data frame, randomly shuffles the data, splits the data from 0 to a then a to b then b to end.
    if a + b + c != 1:
        return 
    indexes = [int(a*len(data_frame_features)), int((a+b)*len(data_frame_features))]
    return np.split(data_frame_features.sample(frac=1, random_state=42),indexes) + np.split(data_frame_labels.sample(frac=1, random_state=42),indexes)

protein_training_X, protein_dev_X, protein_test_X, protein_training_y, protein_dev_y, protein_test_y = split_data(scaled_data, protein_dt_label, 0.7, 0.15, 0.15)

import tensorflow as tf
protein_tensor_features_training = tf.convert_to_tensor(np.array(protein_training_X))
protein_tensor_label_training = tf.convert_to_tensor(np.array(protein_training_y))

protein_tensor_features_dev = tf.convert_to_tensor(np.array(protein_dev_X))
protein_tensor_label_dev = tf.convert_to_tensor(np.array(protein_dev_y))

protein_tensor_features_test = tf.convert_to_tensor(np.array(protein_test_X))
protein_tensor_label_test = tf.convert_to_tensor(np.array(protein_test_y))

scaled_data_tensor = tf.convert_to_tensor(np.array(scaled_data))
protein_dt_label_tensor = tf.convert_to_tensor(np.array(protein_dt_label))

# Construct baseline neural net
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=2,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(protein_tensor_features_training, protein_tensor_label_training, batch_size=10, epochs=10, verbose=0)
predicted_class = np.where(model.predict(protein_tensor_features_test) > 0.5, 1,0)

# testing baseline model 
outfile = open('neural_net_out.txt','w')
outfile.write('Neural Net - 2  Features - Baseline\n')
outfile.write('accuracy: ' + str(sk.accuracy_score(protein_tensor_label_test, predicted_class))+'\n')
outfile.write('f1: ' + str(sk.f1_score(protein_tensor_label_test, predicted_class, average='binary'))+'\n')
outfile.write('precision: ' + str(sk.precision_score(protein_tensor_label_test, predicted_class, average='binary'))+'\n')
outfile.write('recall: ' + str(sk.recall_score(protein_tensor_label_test, predicted_class, average='binary'))+'\n')

predicted_class = np.where(model.predict(scaled_data_tensor) > 0.5, 1,0)
outfile.write('Neural Net - 2  Features - Baseline - Testing on the Entire Dataset\n')
outfile.write('accuracy: ' + str(sk.accuracy_score(protein_dt_label_tensor, predicted_class))+'\n')
outfile.write('f1: ' + str(sk.f1_score(protein_dt_label_tensor, predicted_class, average='binary'))+'\n')
outfile.write('precision: ' + str(sk.precision_score(protein_dt_label_tensor, predicted_class, average='binary'))+'\n')
outfile.write('recall: ' + str(sk.recall_score(protein_dt_label_tensor, predicted_class, average='binary'))+'\n')


# grid search on the batch size, epochs, learning rate
from tensorflow.keras import optimizers
result = [0,0,0,0]
for i in [1,2,4,8,32]:
    for k in [1,2,4,8,32]:
            for j in [0.001,0.01,0,10000]:
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Dense(units=2,activation='relu'))
                model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
                model.compile(optimizer=optimizers.RMSprop(lr=j),loss='binary_crossentropy',metrics=['accuracy'])
                model.fit(protein_tensor_features_training, protein_tensor_label_training, batch_size=k, epochs=i, verbose=0)
                results = model.evaluate(protein_tensor_features_dev, protein_tensor_label_dev)
                if result[3] < results[1]:
                    result[0],result[1],result[2], result[3] = i,k,j,results[1]
            
print(result)

# construct model using the hyperparameters from grid search
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=2,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=result[2]),loss='binary_crossentropy',metrics=['accuracy'])

model.fit(protein_tensor_features_training, protein_tensor_label_training, batch_size=result[1], epochs=result[0], verbose=0)

predicted_class = np.where(model.predict(protein_tensor_features_test) > 0.5, 1,0)


# using the most accurate model via grid search 
outfile.write('Neural Net - 2  Features - Final\n')
outfile.write('accuracy: ' + str(sk.accuracy_score(protein_tensor_label_test, predicted_class))+'\n')
outfile.write('f1: ' + str(sk.f1_score(protein_tensor_label_test, predicted_class, average='binary'))+'\n')
outfile.write('precision: ' + str(sk.precision_score(protein_tensor_label_test, predicted_class, average='binary'))+'\n')
outfile.write('recall: ' + str(sk.recall_score(protein_tensor_label_test, predicted_class, average='binary'))+'\n')

predicted_class = np.where(model.predict(scaled_data_tensor) > 0.5, 1,0)
outfile.write('Neural Net - 2  Features - Final - Testing on the Entire Dataset\n')
outfile.write('accuracy: ' + str(sk.accuracy_score(protein_dt_label_tensor, predicted_class))+'\n')
outfile.write('f1: ' + str(sk.f1_score(protein_dt_label_tensor, predicted_class, average='binary'))+'\n')
outfile.write('precision: ' + str(sk.precision_score(protein_dt_label_tensor, predicted_class, average='binary'))+'\n')
outfile.write('recall: ' + str(sk.recall_score(protein_dt_label_tensor, predicted_class, average='binary'))+'\n')

outfile.close()



