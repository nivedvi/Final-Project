import pickle
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import os

# Load the features dictionary
file_path = "features.pkl"
with open(file_path, 'rb') as file:
    features = pickle.load(file)

# Save the frequency features dictionary
freq_features_dict = features[1]

# Extract people and movements directly from freq_features_dict
people = list(freq_features_dict.keys())

# Zero padding function
def zero_pad(array, max_length):
    pad_len = max_length - len(array)
    if pad_len > 0:
        return np.pad(array, (0, pad_len), 'constant', constant_values=(0,))
    return array

# Precompute the maximum segment length
max_len_segment = max(
    len(segment)
    for person in people
    for movement_name in freq_features_dict[person]['stft'].keys()
    for segment in freq_features_dict[person]['stft'][movement_name]
)

# Process segments for a single person
def process_segments(person, freq_features_dict, max_len_segment):
    rows = []
    movements = freq_features_dict[person]['stft'].keys()
    for movement_name in movements:
        segments_list = freq_features_dict[person]['stft'][movement_name]
        padded_segments = [zero_pad(abs(segment), max_len_segment) for segment in segments_list]
        for segment in padded_segments:
            row = {f'element_{i+1}': element for i, element in enumerate(segment)}
            row['label'] = movement_name
            rows.append(row)
    data_matrix = pd.DataFrame(rows)
    return data_matrix

# Save data matrix for a single person
def save_data_matrix(person, freq_features_dict, max_len_segment, output_dir):
    data_matrix = process_segments(person, freq_features_dict, max_len_segment)
    file_name = os.path.join(output_dir, f"{person}_data_matrix.pkl")
    data_matrix.to_pickle(file_name)
    print(f"Saved {file_name}")
    return data_matrix

# Create a directory to store the files
output_dir = "data_matrices"
os.makedirs(output_dir, exist_ok=True)

# Display available persons
print("Available persons:")
for i, person in enumerate(people):
    print(f"{i}: {person}")

# Get user input to select persons
people_indices = input("Enter the numbers of the persons to create data matrices for (space separated): ").split()
selected_persons = [people[int(index)] for index in people_indices]

# Process and save data matrices for selected persons
data_matrix = Parallel(n_jobs=-1)(delayed(save_data_matrix)(person, freq_features_dict, max_len_segment, output_dir) for person in selected_persons)
