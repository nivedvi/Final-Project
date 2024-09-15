# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:37:16 2024

@author: nived
"""

import classification_new
import numpy as np
from results_analysis import plot_results, save_results, save_figures, save_std_results,save_confusion_matrices
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder

def cross_validation(option=1, person_index=None, method=None, folds=10):
    results_list = []
    matrices = []
    metrics_list = []

    # Initialize the accumulators for average calculation
    ave_conf_matrix = None
    ave_results = None
    ave_metrics = None

    # Initialize the sum of squares of differences for standard deviation calculation
    sq_diff_conf_matrix = None
    sq_diff_results = None
    sq_diff_metrics = None

    for i in range(folds):
        results, details, fig, metrics, confs = classification_new.main(option, person_index, method)
        
        results_list.append(results.iloc[:, :4])
        matrices.append(confs)
        metrics_list.append(metrics)
        
        if ave_conf_matrix is None:
            ave_conf_matrix = {k: np.zeros_like(confs[k]) for k in confs}
            ave_results = pd.DataFrame(0, index=results.index, columns=results.columns[1:4])
            ave_metrics = pd.DataFrame(0, index=metrics.index, columns=metrics.columns)
            sq_diff_conf_matrix = {k: np.zeros_like(confs[k]) for k in confs}
            sq_diff_results = pd.DataFrame(0, index=results.index, columns=results.columns[1:4])
            sq_diff_metrics = pd.DataFrame(0, index=metrics.index, columns=metrics.columns)
        
        ave_conf_matrix = {k: ave_conf_matrix[k] + confs[k] for k in ave_conf_matrix}
        ave_results += results.iloc[:, 1:4]
        ave_metrics += metrics.iloc[:, :]
    
    # Calculate the average
    ave_conf_matrix = {k: ave_conf_matrix[k] / folds for k in ave_conf_matrix}
    ave_results = np.round(ave_results / folds, 4)
    ave_metrics = np.round(ave_metrics / folds, 4)
    ave_conf_matrix = {k: np.round(ave_conf_matrix[k], 1) for k in ave_conf_matrix}
    
    # Calculate the sum of squares of differences from the mean for std deviation
    for i in range(folds):
        confs = matrices[i]
        results = results_list[i].iloc[:, 1:4]
        metrics = metrics_list[i]

        sq_diff_conf_matrix = {k: sq_diff_conf_matrix[k] + (confs[k] - ave_conf_matrix[k])**2 for k in confs}
        sq_diff_results += (results - ave_results)**2
        sq_diff_metrics += (metrics - ave_metrics)**2
    
    # Calculate the standard deviation
    std_conf_matrix = {k: np.sqrt(v / (folds - 1)) for k, v in sq_diff_conf_matrix.items()}
    std_results = np.sqrt(sq_diff_results / (folds - 1))
    std_metrics = np.sqrt(sq_diff_metrics / (folds - 1))
    std_results = np.round(std_results, 4)
    std_metrics = np.round(std_metrics, 4)
    std_conf_matrix = {k: np.round(std_conf_matrix[k], 1) for k in std_conf_matrix}

    # Prepare the average and std dataframes
    ave_conf_matrix = pd.DataFrame({'Confusion Matrix': pd.Series(ave_conf_matrix)})
    std_conf_matrix = pd.DataFrame({'Confusion Matrix Std': pd.Series(std_conf_matrix)})
    
    ave_results.index = ave_conf_matrix.index
    ave_results = pd.concat([ave_results, ave_conf_matrix], axis=1)

    std_results.index = ave_conf_matrix.index
    std_results = pd.concat([std_results, std_conf_matrix], axis=1)
    averages = (ave_results, ave_metrics)
    st_deviations = (std_results, std_metrics)
    return averages, st_deviations, matrices, fig, details

    
def main():
    
    
    for i in range(7):
        
        averages,st_deviations,matrices,fig,details = cross_validation(1, i, 'lda')
        ave_results,ave_metrics = averages
        std_results,std_metrics = st_deviations
        comparison, name, method = details
        if comparison == 1:
            comparison = 'Single'
        elif comparison == 3:
            comparison = 'All'
        
        # Save results and metrics
        #save_results(ave_results, ave_metrics, name)
        save_std_results(std_results, std_metrics, name)
        #save_confusion_matrices(matrices, name)
        
        # Plot and save figures
        #fig_bar, fig_metrics = plot_results(ave_results, ave_metrics, name)
        #save_figures(fig_bar, fig_metrics, fig, name)
        
    # Process all data combined
    averages,st_deviations,matrices,fig,details = cross_validation(1, i, 'lda')
    ave_results,ave_metrics = averages
    std_results,std_metrics = st_deviations
    comparison, name, method = details
    combined_name = "All"
    
    # Save results and metrics
    #save_results(ave_results, ave_metrics, combined_name)
    save_std_results(std_results, std_metrics, combined_name)
    #save_confusion_matrices(matrices, combined_name)
    # Plot and save figures
    #fig_bar, fig_metrics = plot_results(ave_results, ave_metrics, combined_name)
    #save_figures(fig_bar, fig_metrics, fig, combined_name)
    
    
    
    dir_path = 'data_matrices'
    names = ['jordan','laurence','orit','simon','yuval','All']
    variables = {}
    for name in names:
        file_path = f'{dir_path}\\{name}_data_matrix.pkl'
        file_path_lda = f'{dir_path}\\{name}_LDA_data_matrix.pkl'
        files = [file_path,file_path_lda]
        
        for path in files:
            if 'LDA' in path.split('_'):
                with open(path,'rb') as file:
                    globals()[f'df_{name}_lda'] = pickle.load(file).loc[:,['lda_1','lda_2','lda_3','label']]
                    variables[f'df_{name}_lda'] = globals()[f'df_{name}_lda']
            else:
                with open(path,'rb') as file:
                    globals()[f'df_{name}'] =  pickle.load(file).loc[:,['element_1','element_2','element_3','label']]
                    variables[f'df_{name}'] = globals()[f'df_{name}']
                    
     
    label_encoder = LabelEncoder()            
    for frame_name, frame in variables.items():
        fig = plt.figure(figsize=(9,7.2))
        ax = fig.add_subplot(111, projection='3d')
        x = frame.iloc[:, 0]
        y = frame.iloc[:, 1]
        z = frame.iloc[:, 2]
        labels = frame.iloc[:, 3]
        labels_encoded = label_encoder.fit_transform(labels)
        scatter = ax.scatter(x, y, z, c=labels_encoded, cmap='inferno', label=labels , s=2)
        handles, _ = scatter.legend_elements()
        unique_labels = labels.unique()
        legend_labels = [label for label in unique_labels]
        ax.legend(handles, legend_labels, title="Labels")
    
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_zlabel('LD3')
        plt.title(f'3D Scatter Plot for {frame_name}')
        plt.savefig(f'figures\\{frame_name}.png')
       
        plt.show()
        
       
       
       
if __name__ == '__main__':
    main()
