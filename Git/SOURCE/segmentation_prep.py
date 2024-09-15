# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:29:46 2024

@author: nived
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import Person
from skimage.filters import threshold_otsu
from scipy.signal import find_peaks, savgol_filter
from functions_for_project import *
import statistics
from scipy.io import savemat
import pandas as pd

file_path = "persons_data_thresholded.pkl"

persons = pickle_file_r_w(file_path, 'r')

print("Person data loaded from file:", file_path)
for p in persons:
    variable_name = f'p_{p.name}'
    globals()[variable_name] = p
    

original_signals = {}

for p in persons:
    name = p.name
    measurements = p.measurements
    original_signals[name] = measurements;
    
file_path = "signal_dictionary.pkl"

with open(file_path,'wb') as file:
    pickle.dump(original_signals,file)