# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:00:29 2024

@author: nived
"""


import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from Person import Person
from skimage.filters import threshold_otsu
from scipy.signal import find_peaks, savgol_filter 
from functions_for_project import *

    
file_path = "persons_data.pkl"

persons = pickle_file_r_w(file_path, 'r')


print("Person data loaded from file:", file_path)
for p in persons:
    variable_name = f'p_{p.name}'
    globals()[variable_name] = p
    
   

for p in persons:
   for key in p.measurements:
       p.measurements[key] = np.transpose(p.measurements[key])
   transposed = p.is_transposed("measurements")
   print(f"{p.name}'s {transposed[1]} is transposed = {transposed[0]}")        
            

                
           
#the function is defined again here since using it from a different module casues problems

def filter_matrices(filter_name, rewrite = False, *args, **kwargs):
    import copy
    filter_args = args
    filter_kwargs = kwargs
    for p in persons:
        print(f'{p.name}:')
        if rewrite == True:
            #p.filt_measure = {}
            for key, matrix in p.measurements.items():
                # Deep copy the original matrix to avoid modifying p.measurements
                mat_copy = copy.deepcopy(np.abs(matrix))
                # Apply Savitzky-Golay filter to each row of the copied matrix
                filtered_matrix = np.array([filter_name(row ,*filter_args, **filter_kwargs) for row in mat_copy])
                # Store the filtered matrix in p.filt_measure
                p.filt_measure[key] = filtered_matrix
                print(f"    {key}: *Filtered* -- overwritten")
        else:
            for key, matrix in p.filt_measure.items():
                # Deep copy the original matrix to avoid modifying p.measurements
                mat_copy = copy.deepcopy(matrix)
                # Apply Savitzky-Golay filter to each row of the copied matrix
                filtered_matrix = np.array([filter_name(row ,*filter_args, **filter_kwargs) for row in mat_copy])
                # Store the filtered matrix in p.filt_measure
                p.filt_measure[key] = filtered_matrix
                print(f"    {key}: *Filtered*")

            
# 3 time filtering
filter_matrices(savgol_filter, rewrite=True, window_length = 1001,polyorder=4)
filter_matrices(savgol_filter, rewrite=False, window_length = 1001,polyorder=4)      
filter_matrices(savgol_filter, rewrite=False, window_length = 1001,polyorder=4)  


     


file_path = "persons_data_thresholded.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(persons, file)
