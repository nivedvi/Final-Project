import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_results(df, metrics_df, name):
    # Transpose DataFrame for plotting
    df_forplot = df.iloc[:, 0:3].T
    models = list(df.index)

    # Plot the original DataFrame
    fig, ax = plt.subplots(figsize=(12, 7.2)) 
    df_forplot.plot(kind='bar', ax=ax)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Values')
    ax.set_title(f'Results of {name.capitalize()}, 4 Models')
    ax.grid()
    ax.legend(models, title='Models')
    plt.show()
    
    # Plot the combined metrics DataFrame grouped by model
    grouped_metrics_df = pd.DataFrame()
    for model in models:
        model_metrics = metrics_df.filter(like=model, axis=1)
        grouped_metrics_df[model] = model_metrics.mean(axis=1)

    move_fig, move_ax = plt.subplots(figsize=(12, 7.2))
    grouped_metrics_df.plot(kind='bar', ax=move_ax)
    move_ax.set_xlabel('Classes')
    move_ax.set_ylabel('Values')
    move_ax.set_title(f'Performance Metrics for {name.capitalize()} Across Movements')
    move_ax.grid()
    move_ax.legend(title='Models')
    plt.show()
    
    return fig, move_fig

def save_confusion_matrices(dict_list, name):
    # Create the base directory for results
    base_dir = os.path.join('results', name, 'Confusion Matrices')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for i, conf_dict in enumerate(dict_list):
        file_name = f'Confusion_Matrix_{i+1}.xlsx'
        file_path = os.path.join(base_dir, file_name)
        
        with pd.ExcelWriter(file_path) as writer:
            for key, matrix in conf_dict.items():
                df = pd.DataFrame(matrix)
                df.to_excel(writer, sheet_name=key)
                
        print(f'Saved {file_name} in {base_dir}')


def save_std_results(std_df, std_metrics_df, name):
    # Save the standard deviation results and metrics DataFrame to Excel
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    dir_path = os.path.join(results_dir, name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    std_df.to_excel(os.path.join(dir_path, f'Std_Results_for_{name}.xlsx'))
    std_metrics_df.to_excel(os.path.join(dir_path, f'Std_Metrics_for_{name}.xlsx'))

def save_results(df, metrics_df, name):
    # Save the results and metrics DataFrame to Excel
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    dir_path = os.path.join(results_dir, name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    df.to_excel(os.path.join(dir_path, f'Results_for_{name}.xlsx'))
    metrics_df.to_excel(os.path.join(dir_path, f'Metrics_for_{name}.xlsx'))

def save_figures(fig, move_fig, roc_fig, name):
    # Save the figures to the 'figures' directory
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    dir_path = os.path.join(figures_dir, name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    fig.savefig(os.path.join(dir_path, f'Results_Plot_for_{name}.png'))
    move_fig.savefig(os.path.join(dir_path, f'Metrics_Plot_for_{name}.png'))
    roc_fig.savefig(os.path.join(dir_path, f'ROC_Curve_for_{name}.png'))

print("Processing complete.")
