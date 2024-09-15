import pickle
import os
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

persons = ['emma', 'jordan', 'laurence', 'niv', 'orit', 'simon', 'yuval']

def apply_lda(data_matrix, desired_n_components=50):
    """
    Applies LDA to the given data matrix with a desired number of components.

    Parameters:
    data_matrix (pd.DataFrame): The input data matrix.
    desired_n_components (int): The desired number of components to keep.

    Returns:
    pd.DataFrame: DataFrame with reduced features after applying LDA.
    """
    # Separate features and labels
    X = data_matrix.drop(columns=['label'])
    y = data_matrix['label']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Determine the maximum number of components
    n_features = X_scaled.shape[1]
    n_classes = len(np.unique(y))
    n_components = min(desired_n_components, n_features, n_classes - 1)
    
    if n_components < desired_n_components:
        print(f"Warning: Unable to reduce to {desired_n_components} components. Using {n_components} components instead.")
    
    # Apply LDA
    lda = LDA(n_components=n_components)
    X_reduced = lda.fit_transform(X_scaled, y)

    # Create a DataFrame with the reduced features
    reduced_data_matrix = pd.DataFrame(X_reduced, columns=[f'lda_{i+1}' for i in range(X_reduced.shape[1])])
    reduced_data_matrix['label'] = y.values
    
    return reduced_data_matrix

# Load the data matrices for all persons
data_matrices_dir = "data_matrices"
person_files = [f for f in os.listdir(data_matrices_dir) if f.endswith('_data_matrix.pkl') and 'reduced' not in f.split('_')]

# Dictionary to store data matrices for each person
data_matrices = {}

# Load each person's data matrix and save it as a variable
for person in persons:
    file_path = os.path.join(data_matrices_dir, f"{person}_data_matrix.pkl")
    with open(file_path, 'rb') as file:
        data_matrix = pickle.load(file)
        data_matrices[person] = data_matrix
        print(f"Data matrix for {person} loaded and saved as a variable.")

# Concatenate all individual data matrices into one DataFrame
data_frames = [data_matrices[person] for person in persons]
combined_data_matrix = pd.concat(data_frames, ignore_index=True)

# Apply LDA to each person's data matrix and save the reduced data matrix
for person, data_matrix in data_matrices.items():
    reduced_data_matrix = apply_lda(data_matrix, desired_n_components=50)  # Ensure 50 components if possible
    reduced_data_matrix_file = f"{person}_LDA_data_matrix.pkl"
    reduced_data_matrix.to_pickle(os.path.join(data_matrices_dir, reduced_data_matrix_file))
    print(f"Reduced data matrix for {person} saved to {reduced_data_matrix_file}")

# Apply LDA to the combined data matrix and save the reduced data matrix
reduced_combined_data_matrix = apply_lda(combined_data_matrix, desired_n_components=50)  # Ensure 50 components if possible
combined_reduced_data_matrix_file = "combined_LDA_data_matrix.pkl"
reduced_combined_data_matrix.to_pickle(os.path.join(data_matrices_dir, combined_reduced_data_matrix_file))
print(f"Combined reduced data matrix saved to {combined_reduced_data_matrix_file}")
