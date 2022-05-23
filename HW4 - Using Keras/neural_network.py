#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 13:58:54 2021

@author: chrisnguyen
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

protein_dt = pd.read_csv('herg_features.tsv',sep='\t',skiprows=(1))

protein_columns = protein_dt.columns
protein_dt_features = protein_dt[protein_columns[1:11]]
#protein_dt_label = protein_dt[protein_columns[11]]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(protein_dt_features),columns=protein_columns[1:11])

import keras