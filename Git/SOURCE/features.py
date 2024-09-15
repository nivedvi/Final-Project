import numpy as np
import pickle
from scipy.signal import stft
import copy

def get_user_choice():
    while True:
        choice = input("Choose the signal row (1, 2, or 3) 'all' for all: ")
        if choice in ['1', '2', '3' ]:
            return int(choice) - 1
        elif choice.lower() == 'all':
            print(range(3))
            return range(3)
        print("Invalid choice, please choose 1, 2, or 3.")

# Load segments from segments.pkl
file_path = "segments.pkl"
with open(file_path, 'rb') as file:
    segments = pickle.load(file)

print("Segments data loaded from file:", file_path)

# Get user choice for the signal row once
chosen_row = get_user_choice()
#print(f"You have chosen signal row {chosen_row + 1}")

# Initialize an empty dictionary to store the time features
time_features = {}

# Iterate over the segments dictionary
for name in segments:
    # Create a dictionary for the person's time features if it doesn't exist
    if name not in time_features:
        time_features[name] = {}

    for move in segments[name]:
        segments_list = segments[name][move]

        # Check if the segmented signal is not None
        if segments_list:
            # Create a list to store the features for each segment
            move_list = []

            # Iterate over each segment in the segmented signal
            for seg in segments_list:
                # Extract the chosen row
                row = seg[chosen_row]

                # Example feature extraction (mean value)
                feature_value = np.mean(row)

                # Append the feature value to the move list
                move_list.append(feature_value)

            # Store the move list in the time features dictionary
            time_features[name][move] = move_list

# Create a deep copy of time_features to prevent changes to the original object
time_features_copy = copy.deepcopy(time_features)

# Sampling frequency
fs = 1000  
nperseg = 256
noverlap = 128

# Dictionary to store frequency features
frequency_features = {}

# Iterate over the segments dictionary
for name in segments:
    # Create a dictionary for the person's frequency features if it doesn't exist
    if name not in frequency_features:
        frequency_features[name] = {'stft': {}}  # Create the 'stft' dictionary here
        
    # Iterate over each movement for the current person
    for move in segments[name]:
        segments_list = segments[name][move]

        # Check if the segmented signal for the current movement is not None
        if segments_list:
            # Create a list to store the features for each segment
            move_list = []

            # Iterate over each segment in the segmented signal
            for seg in segments_list:
                # Extract the chosen row
                row = seg[chosen_row]
                nperseg = 256
                len_row = max(np.shape(row))
                # Check if the row length is sufficient for the STFT parameters
                if len_row >= nperseg:
                    # Apply STFT to calculate the feature
                    f, t, feature_value = stft(row, fs, nperseg=nperseg, noverlap=64)
                    
                    # Flatten the STFT matrix into a single vector
                    feature_value_flat = feature_value.flatten()

                    # Append the flattened feature to the move list
                    move_list.append(feature_value_flat)
                else:
                    # If the segment is too short, skip it or handle it accordingly
                    continue

            # Store the move list in the frequency_features dictionary
            frequency_features[name]['stft'][move] = move_list

# Make a deep copy of frequency_features
frequency_features_copy = copy.deepcopy(frequency_features)

# Save the extracted features to a file
output_file_path = "features.pkl"
with open(output_file_path, 'wb') as file:
    pickle.dump((time_features, frequency_features), file)

print("Feature extraction completed and saved to file:", output_file_path)
