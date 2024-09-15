import os
import pickle
import pandas as pd
import numpy as np

# Directory paths
main_directory = '.'
code_directory = os.path.join(main_directory, 'code')

# Load the signal_dictionary
signal_dict_path = os.path.join(main_directory, 'signal_dictionary.pkl')
with open(signal_dict_path, 'rb') as file:
    signal_dictionary = pickle.load(file)

# List of persons
persons = list(signal_dictionary.keys())

# Function to load time stamps from code directory
def load_time_stamps(person):
    file_path = os.path.join(code_directory, f'{person}.txt')
    return pd.read_csv(file_path, sep='\t').apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Load time stamps for each person and organize into segment_stamps dictionary
segment_stamps = {}
for person in persons:
    time_stamps_df = load_time_stamps(person)
    time_stamps_df = time_stamps_df.iloc[1:, 1:].reset_index(drop=True)  # Drop the first row which contains the titles for start and end and reset the index
    
    person_stamps = {}
    for col in range(0, time_stamps_df.shape[1], 2):  # Iterate through the pairs of columns
        
        movement = time_stamps_df.columns[col].strip()
        print(movement)
        start_end_matrix = np.array(time_stamps_df.iloc[:, col:col + 2].values.T, dtype=float)  # Ensure float type
        start_end_matrix = (start_end_matrix * 1000).astype(int)  # Convert times to indices
        person_stamps[movement] = start_end_matrix
    
    segment_stamps[person] = person_stamps

# Initialize the segments dictionary
segments = {}

# Slice each signal according to the segment stamps
for person, movements in segment_stamps.items():
    person_signal_dict = signal_dictionary[person]
    person_segments = {}
    
    for movement, start_end_matrix in movements.items():
        movement_signal = np.array(person_signal_dict[movement])
        movement_segments = []
        
        for start_idx, end_idx in zip(start_end_matrix[0], start_end_matrix[1]):
            segment = movement_signal[:, start_idx:end_idx]  # Slice the signal
            movement_segments.append(segment)
            
        person_segments[movement] = movement_segments
    segments[person] = person_segments
        
file_path = 'segments.pkl'
with open(file_path,'wb') as file:
    pickle.dump(segments,file)