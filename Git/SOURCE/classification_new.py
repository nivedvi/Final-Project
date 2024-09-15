import pickle
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from model_voting import train_evaluate_voting

def load_data_matrix(person, method):
    if method == 'pca':
        file_suffix = '_PCA_data_matrix.pkl'
    elif method == 'lda':
        file_suffix = '_LDA_data_matrix.pkl'
    else:
        file_suffix = '_data_matrix.pkl'
    
    file_path = os.path.join('data_matrices', f"{person}{file_suffix}")
    with open(file_path, 'rb') as file:
        data_matrix = pickle.load(file)
    return data_matrix

def split_data(data_matrix):
    X = data_matrix.drop(columns=['label'])
    y = data_matrix['label']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)
    return X_train, X_test, y_train, y_test, label_encoder

def main(option=None, person_index=None, method_choice=None):
    persons = ['emma', 'jordan', 'laurence', 'niv', 'orit', 'simon', 'yuval']
    
    if option is None:
        option = int(input("Choose an option (1: Single person, 2: Single person vs All the rest, 3: All data): ").strip())

    if option == 1:
        if person_index is None:
            print("Available persons:")
            for i, person in enumerate(persons):
                print(f"{i}: {person}")
            person_index = int(input("Enter the number of the person to use: ").strip())
        selected_person = persons[person_index]
        if method_choice is None:
            method_choice = input("Enter 'pca' to use PCA, 'lda' to use LDA, or 'original' to use the original data: ").strip().lower()
        data_matrix = load_data_matrix(selected_person, method_choice)
        X_train, X_test, y_train, y_test, label_encoder = split_data(data_matrix)
        le = label_encoder
        label_mapping = {index: label for index, label in enumerate(le.classes_)}
        with open('label_mapping.pkl','wb') as file:
            pickle.dump(label_mapping, file)
    elif option == 2:
        if person_index is None:
            print("Available persons:")
            for i, person in enumerate(persons):
                print(f"{i}: {person}")
            person_index = int(input("Enter the number of the person to use as test set: ").strip())
        test_person = persons[person_index]
        if method_choice is None:
            method_choice = input("Enter 'pca' to use PCA, 'lda' to use LDA, or 'original' to use the original data: ").strip().lower()
        test_data_matrix = load_data_matrix(test_person, method_choice)
        train_data_frames = [load_data_matrix(person, method_choice) for i, person in enumerate(persons) if i != person_index]
        train_data_matrix = pd.concat(train_data_frames, ignore_index=True)
        X_train = train_data_matrix.drop(columns=['label'])
        y_train = train_data_matrix['label']
        X_test = test_data_matrix.drop(columns=['label'])
        y_test = test_data_matrix['label']
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

    elif option == 3:
        if method_choice is None:
            method_choice = input("Enter 'pca' to use PCA, 'lda' to use LDA, or 'original' to use the original data: ").strip().lower()
        data_frames = [load_data_matrix(person, method_choice) for person in persons]
        combined_data_matrix = pd.concat(data_frames, ignore_index=True)
        X_train, X_test, y_train, y_test, label_encoder = split_data(combined_data_matrix)

    else:
        print("Invalid option. Exiting.")
        return

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    results, fig,  = train_evaluate_voting(X_train_scaled, y_train, X_test_scaled, y_test, label_encoder)

    results_summary = []
    metrics_df_list = []
    conf_matrices = {}
    for model_name, result in results.items():
        y_test, y_pred, overall_auc, auc, fscore_per_class, accuracy_per_class, fscore, accuracy, mse, conf_matrix, roc_auc = result
        df_results = pd.DataFrame({'Actual': label_encoder.inverse_transform(y_test), 'Predicted': label_encoder.inverse_transform(y_pred)})

        auc_df = pd.DataFrame.from_dict(auc, orient='index', columns=[f'{model_name}_AUC'])
        fscore_df = pd.DataFrame.from_dict(fscore_per_class, orient='index', columns=[f'{model_name}_F1'])
        accuracy_df = pd.DataFrame.from_dict(accuracy_per_class, orient='index', columns=[f'{model_name}_Accuracy'])
        
        metrics_df = pd.concat([auc_df, fscore_df, accuracy_df], axis=1)
        metrics_df_list.append(metrics_df)

        summary = {
            'Model': model_name,
            'Overall AUC': overall_auc,
            'F-score': fscore,
            'Accuracy': accuracy,
            #'Mean Squared Error': mse,
            'Confusion Matrix': conf_matrix,
            'Results DataFrame': df_results
        }
        results_summary.append(summary)
        conf_matrices[model_name] = conf_matrix
    results_summary_df = pd.DataFrame(results_summary).iloc[:,:53]
 
    for index, row in results_summary_df.iterrows():
        print(f"\nResults for {row['Model']}:")
        print(f"Overall AUC: {row['Overall AUC']}")
        print(f"F-score: {row['F-score']:.4f}")
        print(f"Accuracy: {row['Accuracy']:.4f}")
        #print(f"Mean Squared Error: {row['Mean Squared Error']:.4f}")
        print("Confusion Matrix:\n", row['Confusion Matrix'])

    metrics_df = pd.concat(metrics_df_list, axis=1)
    metrics_df.index.name = 'Class'

    return results_summary_df, (option, selected_person if option != 3 else 'all', method_choice), fig, metrics_df,conf_matrices

if __name__ == '__main__':
    # Optionally provide parameters for testing
    option = None
    person_index = None
    method_choice = None
    main(option, person_index, method_choice)
