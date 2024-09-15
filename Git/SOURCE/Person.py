# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 21:03:16 2024

@author: nived
"""


import numpy as np
import matplotlib.pyplot as plt

class Person:
    def __init__(self, name, data_matrix, time_vec):
        
        move_mat = {'flexion':None, 'extension':None, 'ulnar_deviation':None, 'radial_deviation':None,
                             'hook':None, 'power':None, 'spherical':None, 'precision':None, 'lateral':None,
                             'pinch':None}
        self.name = name
        self.data_matrix = data_matrix
        self.time = time_vec
        self.timestamps_list = []
        self.measurements = {'flexion':None, 'extension':None, 'ulnar_deviation':None, 'radial_deviation':None,
                             'hook':None, 'power':None, 'spherical':None, 'precision':None, 'lateral':None,
                             'pinch':None}
        self.thresholds = {'flexion':None, 'extension':None, 'ulnar_deviation':None, 'radial_deviation':None,
                             'hook':None, 'power':None, 'spherical':None, 'precision':None, 'lateral':None,
                             'pinch':None}
        self.segments = {}
        
        self.segments = {}
        self.time_domain = {'MAV': None, 'VAR': None, 'energy': None, 'WL': None,'RMS': None, 'skew':None, 
                            'kurtosis': None }
        self.segmented_signal = {}
        for key in self.time_domain:
            self.time_domain[key] = move_mat
        self.isTimeFeatSeg = False
    
    def is_transposed(self, attribute):
        if hasattr(self, attribute):
            matrix = getattr(self, attribute)
            if isinstance(matrix, dict):
                return all(len(value) < len(value[0]) for value in matrix.values()) , attribute
            else:
                raise TypeError("Attribute must be a dictionary containing matrices")
        else:
            raise AttributeError(f"'{attribute}' attribute not found")
    

        





