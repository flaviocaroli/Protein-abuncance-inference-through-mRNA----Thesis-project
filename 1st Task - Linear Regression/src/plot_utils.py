import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import mode


def plot_prot_heatmap(data, protein_col_start=7):
    expression_data = data.iloc[:, protein_col_start:]
    plt.figure(figsize=(20, 10))
    sns.heatmap(expression_data.T, cmap='viridis', cbar=True)
    plt.xlabel('Samples')
    plt.ylabel('Proteins')
    plt.title('Heatmap of Protein Expression')
    plt.show()

# Function to plot correlation matrix heatmap of proteins
def plot_correlation_heatmap(data, protein_col_start=7):
    expression_data = data.iloc[:, protein_col_start:]
    correlation_matrix = expression_data.corr()
    plt.figure(figsize=(20, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True, annot=True, fmt=".2f")
    plt.title('Correlation Matrix Heatmap of Proteins')
    plt.show()
def density_plot(df, palette):
    values = df.values.flatten()
    sns.kdeplot(values, fill=True, palette = palette)
    plt.title('Density Plot of Data Values')
    plt.show()
def plot_mse_distribution(results_df):
    """
    Compute and plot the distribution of MSE across the 10 folds for each protein.
    :param results_df: DataFrame containing the results with cv_mse_r2 for each protein.
    """
    mse_list = []

    for idx, row in results_df.iterrows():
        protein_name = row['protein']
        for fold_mse in row['cv_mse_r2']['mse']:
            mse_list.append({'protein': protein_name, 'mse': fold_mse})

    mse_df = pd.DataFrame(mse_list)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='protein', y='mse', data=mse_df)
    plt.xticks(rotation=90)
    plt.title('MSE Distribution Across 10 Folds for Each Protein')
    plt.xlabel('Protein')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.show()

def plot_r2_distribution(results_df):

    """
    Compute and plot the distribution of R2 across the 10 folds for each protein.
    :param results_df: DataFrame containing the results with cv_mse_r2 for each protein.
    """
    r2_list = []

    for idx, row in results_df.iterrows():
        protein_name = row['protein']
        for fold_r2 in row['cv_mse_r2']['r2']:
            r2_list.append({'protein': protein_name, 'r2': fold_r2})

    r2_df = pd.DataFrame(r2_list)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='protein', y='r2', data=r2_df)
    plt.xticks(rotation=90)
    plt.title('R2 Distribution Across 10 Folds for Each Protein')
    plt.xlabel('Protein')
    plt.ylabel('R2')
    plt.tight_layout()
    plt.show()

def compare_datasets(df1, df2, limit,corr = False, name1='DF1', name2='DF2'):
    values_df1 = df1.values.flatten()
    values_df2 = df2.values.flatten()
    
    combined_df = pd.DataFrame({
        'Values': np.concatenate([values_df1, values_df2]),
        'Dataset': [name1] * len(values_df1) + [name2] * len(values_df2)
    })
    
    
    
    plt.figure(figsize=(8, 6))
    
    # KDE plot
    sns.kdeplot(data=combined_df, x='Values', hue='Dataset', common_norm=False, 
                palette={name1: 'blue', name2: 'orange'}, fill=True)
    if corr == True:
        median_df1 = np.median(values_df1)
        median_df2 = np.median(values_df2)
        plt.axvline(median_df1, color='blue', linestyle='--', linewidth=1, label=f'Median {name1}')
        plt.axvline(median_df2, color='orange', linestyle='--', linewidth=1, label=f'Median {name2}')
    plt.title('')
    plt.xlabel('')  # X-axis label for values
    plt.ylabel('') 
    if limit == 10: # Set x-axis limits
        plt.xlim(-10, 10)
    
    
    plt.tight_layout()
    plt.show()

def compare_datasets2(df1, df2, name1='DF1', name2='DF2'):
    values_df1 = df1.values.flatten()
    values_df2 = df2.values.flatten()
    
    combined_df = pd.DataFrame({
        'Correlations': np.concatenate([values_df1, values_df2]),
        'Dataset': [name1] * len(values_df1) + [name2] * len(values_df2)
    })
    
    median_df1 = np.median(values_df1)
    median_df2 = np.median(values_df2)
    mean_df1 = np.mean(values_df1)
    mean_df2 = np.mean(values_df2)
    
    # Round values to the first decimal digit before computing the mode
    rounded_df1 = np.round(values_df1, 2)
    rounded_df2 = np.round(values_df2, 2)
    
    mode_df1 = mode(rounded_df1).mode
    mode_df2 = mode(rounded_df2).mode

    plt.figure(figsize=(8, 6))
    
    # KDE plot
    sns.kdeplot(data=combined_df, x='Correlations', hue='Dataset', common_norm=False, 
                palette={name1: 'green', name2: 'purple'}, fill=True)
    #plt.axvline(median_df1, color='green', linestyle='--', linewidth=1, label=f'Median {name1}')
    #plt.axvline(median_df2, color='purple', linestyle='--', linewidth=1, label=f'Median {name2}')
    #plt.axvline(mean_df1, color='blue', linestyle='-', linewidth=1, label=f'Mean {name1}')
    #plt.axvline(mean_df2, color='red', linestyle='-', linewidth=1, label=f'Mean {name2}')
    plt.axvline(mode_df1, color='cyan', linestyle=':', linewidth=1, label=f'Mode {name1}')
    plt.axvline(mode_df2, color='magenta', linestyle=':', linewidth=1, label=f'Mode {name2}')
    plt.title('')
    plt.legend()
    plt.ylabel('')
    plt.xlabel('')  
    plt.tight_layout()
    
    print(f'{name1} - Median: {median_df1}, Mean: {mean_df1}, Mode: {mode_df1}')
    print(f'{name2} - Median: {median_df2}, Mean: {mean_df2}, Mode: {mode_df2}')
    
    plt.show()


def compare_datasets3(spearman_corr, pearson_corr, name1='Spearman', name2='Pearson'):
    # Create DataFrame for plotting
    combined_df = pd.DataFrame({
        'Correlation Type': [name1] * len(spearman_corr) + [name2] * len(pearson_corr),
        'Correlation Value': list(spearman_corr.values.flatten()) + list(pearson_corr.values.flatten())
    })
    
    plt.figure(figsize=(8, 6))
    
    # Violin plot for both Spearman and Pearson correlations
    sns.violinplot(data=combined_df, x='Correlation Type', y='Correlation Value', palette={'Spearman': 'purple', 'Pearson': 'green'})
    
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_performance(grid_search):
    results_df = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(10, 6))
    sns.pointplot(x='param_alpha', y='-mean_test_score', hue='param_l1_ratio', data=results_df)
    plt.xlabel('Alpha')
    plt.ylabel('Mean Test Score (Neg MSE)')
    plt.title('Performance of Different Alphas and L1 Ratios')
    plt.legend(title='L1 Ratio')
    plt.show()

def plot_counts_barchart(counts_dict):
    plt.figure(figsize=(12, 7))
    bars = plt.bar(counts_dict.keys(), counts_dict.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # Add numbers on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_mse_histograms(datasets, names):
    colors = ['lightblue', 'orange', 'lightgreen']  # Selected colors for the three datasets
    n_datasets = len(datasets)
    bar_width = 0.25  # Width of each bar
    plt.figure(figsize=(14, 8))

    for i, (enet20_pred, name, color) in enumerate(zip(datasets, names, colors)):
        # Step 1: Split and recognize the tissues
        counts_dict = dict()
        for sample in enet20_pred.columns:
            if '_' not in sample:
                continue
            tissue_name = sample.split('_', 1)[1]
            counts_dict[tissue_name] = counts_dict.get(tissue_name, 0) + 1

        # Step 2: Aggregate values for each tissue and compute the MSE
        mse_dict = dict()
        for tissue in counts_dict:
            tissue_columns = [col for col in enet20_pred.columns if tissue in col]
            all_values = enet20_pred[tissue_columns].values.flatten()
            mse = np.mean(np.square(all_values - np.mean(all_values)))
            mse_dict[tissue] = mse

        # Step 3: Plot the histogram
        tissues = list(mse_dict.keys())
        mse_values = list(mse_dict.values())

        # Offset the bar positions for each dataset
        bar_positions = np.arange(len(tissues)) + i * bar_width

        plt.bar(bar_positions, mse_values, bar_width, alpha=0.7, label=name, color=color)

    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.xticks(np.arange(len(tissues)) + bar_width, tissues, rotation=45, ha='right')  # Rotate x-axis labels and align them to the right
    plt.legend()
    plt.tight_layout()
    plt.show()

  
def densities_alo(datasets):
    """
    Plots the density distributions of the given datasets with their mean, variance, and median highlighted.
    Prints the variance, mean, and median of the datasets and returns a LaTeX table with these metrics.
    
    Parameters:
        datasets (list of pd.DataFrame): List containing the datasets to be plotted.
    """
    metrics = {'Dataset': [], 'Mean': [], 'Variance': [], 'Median': []}
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 8))
    
    if len(datasets) == 1:
        axes = [axes] 
    
    for i, data in enumerate(datasets):
        # Convert to numeric and drop NaNs
        data_numeric = data.apply(pd.to_numeric, errors='coerce').dropna()
        density = data_numeric.values.flatten()
        
        mean = np.mean(density)
        variance = np.var(density)
        median = np.median(density)
        
        metrics['Dataset'].append(f'Dataset {i+1}')
        metrics['Mean'].append(mean)
        metrics['Variance'].append(variance)
        metrics['Median'].append(median)
        
        sns.kdeplot(density, ax=axes[i], color='lightblue')
        axes[i].axvline(mean, color='purple', linestyle='--')
        axes[i].axvline(median, color='g', linestyle='-.')
        
        axes[i].set_xlim(0, 0.5)
        axes[i].legend([f'Mean: {mean:.2f}', f'Variance: {variance:.2f}', f'Median: {median:.2f}'])
    
    plt.tight_layout()
    plt.show()
    
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)


def plot_combined_errors_by_tissue_norm(datasets):
    """
    Plots the combined density distributions of the given datasets with their mean, variance, and median highlighted.
    Prints the variance, mean, and median of the datasets and returns a LaTeX table with these metrics.
    
    Parameters:
        datasets (list of pd.DataFrame): List containing the datasets to be plotted.
    """
    if len(datasets) == 1:
        dataset_names = ['l1 ratio = 0.1']
    else:
        dataset_names = ['l1 ratio = 0.1', 'l1 ratio = 0.5', 'l1 ratio = 0.9']
    
    tissue_sums = {name: {} for name in dataset_names}
    tissue_counts = {name: {} for name in dataset_names}
    
    for name, data in zip(dataset_names, datasets):
        for sample in data.columns:
            if '_' not in sample:
                continue
            tissue_name = sample.split('_', 1)[1]
            tissue_sums[name][tissue_name] = tissue_sums[name].get(tissue_name, 0) + data[sample].sum()
            tissue_counts[name][tissue_name] = tissue_counts[name].get(tissue_name, 0) + 1
    
    #average errors by dividing by the number of cells
    tissue_avgs = {name: {tissue: tissue_sums[name][tissue] / tissue_counts[name][tissue] for tissue in tissue_sums[name]} for name in dataset_names}
    
    tissues = sorted(set(tissue for tissue_avg in tissue_avgs.values() for tissue in tissue_avg))
    x = np.arange(len(tissues))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['magenta', 'lightblue', 'lightgreen']
    
    for i, (name, color) in enumerate(zip(dataset_names, colors[:len(dataset_names)])):
        avgs = [tissue_avgs[name].get(tissue, 0) for tissue in tissues]
        ax.bar(x + i * width, avgs, width, label=name, color=color)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks(x + width / len(dataset_names))
    ax.set_xticklabels(tissues, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
####MSE
def plot_combined_mse_by_tissue_norm(datasets):
    """
    Plots the combined density distributions of the given datasets with their mean squared error (MSE) highlighted.
    Prints the variance, mean, and median of the datasets and returns a LaTeX table with these metrics.
    
    Parameters:
        datasets (list of pd.DataFrame): List containing the datasets to be plotted.
    """
    if len(datasets) == 1:
        dataset_names = ['l1 ratio = 0.1']
    else:
        dataset_names = ['l1 ratio = 0.1', 'l1 ratio = 0.5', 'l1 ratio = 0.9']
    
    tissue_sums = {name: {} for name in dataset_names}
    tissue_counts = {name: {} for name in dataset_names}
    
    for name, data in zip(dataset_names, datasets):
        for sample in data.columns:
            if '_' not in sample:
                continue
            tissue_name = sample.split('_', 1)[1]
            tissue_sums[name][tissue_name] = tissue_sums[name].get(tissue_name, 0) + (data[sample] ** 2).sum()
            tissue_counts[name][tissue_name] = tissue_counts[name].get(tissue_name, 0) + len(data[sample])
    
    # Calculate MSE by dividing the sum of squared errors by the number of cells
    tissue_mses = {name: {tissue: tissue_sums[name][tissue] / tissue_counts[name][tissue] for tissue in tissue_sums[name]} for name in dataset_names}
    
    tissues = sorted(set(tissue for tissue_mse in tissue_mses.values() for tissue in tissue_mse))
    x = np.arange(len(tissues))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['magenta', 'lightblue', 'lightgreen']
    
    for i, (name, color) in enumerate(zip(dataset_names, colors[:len(dataset_names)])):
        mses = [tissue_mses[name].get(tissue, 0) for tissue in tissues]
        ax.bar(x + i * width, mses, width, label=name, color=color)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks(x + width / len(dataset_names))
    ax.set_xticklabels(tissues, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
