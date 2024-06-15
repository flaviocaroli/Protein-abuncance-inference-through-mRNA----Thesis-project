import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import ast

################# Flavio Caroli #################

def get_repeated_sample_names(proteomics_df: pd.DataFrame) -> list:
    """
    Identifies the sample names (part of the column labels before 'TenPx') that appear in more than one Ten-plex in the provided DataFrame.

    Args:
        proteomics_df (pandas.DataFrame): The DataFrame containing the proteomics data.

    Returns:
        list: A list of sample names that appear in more than one Ten-plex.
    """
    # Extract the sample names from the column labels
    sample_names = proteomics_df.columns.str.extract(r'^(.*?)_TenPx', expand=False)
    sample_name_counts = sample_names.value_counts()
    repeated_sample_names = sample_name_counts[sample_name_counts > 1].index.tolist()

    return repeated_sample_names

def plot_top_correlated_genes(gene_df: pd.DataFrame, protein_df: pd.DataFrame, results_df: pd.DataFrame, top_n: int = 5):
    """
    Plots the top N most correlated gene/protein pairs based on R-value in a 5x4 grid.

    Parameters:
    - gene_df (pd.DataFrame): DataFrame containing gene expression levels.
    - protein_df (pd.DataFrame): DataFrame containing protein expression levels.
    - results_df (pd.DataFrame): DataFrame containing the linear regression results.
    - top_n (int): The number of top correlated genes/protein pairs to plot.
    """
    
    # Sort by 'R-Value' in descending order 
    top_genes = results_df.sort_values(by='R-Value', ascending=False).head(top_n).index
    
    
    fig, axes = plt.subplots(nrows=top_n, ncols=1, figsize=(10, 5 * top_n), sharex=True)
    axes = axes.flatten()  
    
    for i, gene in enumerate(top_genes):
        
        if i >= len(axes):
            break
        slope = results_df.loc[gene, 'Slope']
        intercept = results_df.loc[gene, 'Intercept']
        gene_data = gene_df.loc[gene]
        protein_data = protein_df.loc[gene]
        
        sns.scatterplot(x=gene_data, y=protein_data, ax=axes[i], label=f'{gene}')

        x_vals = np.linspace(min(gene_data), max(gene_data), 100)
        y_vals = intercept + slope * x_vals
        sns.lineplot(x=x_vals, y=y_vals, ax=axes[i], color='red')
    
        axes[i].set_title(f'{gene}')
        axes[i].set_xlabel('Gene Expression')
        axes[i].set_ylabel('Protein Expression')
        axes[i].legend()
        
    plt.tight_layout()
    plt.show()

def compute_correlations(dataframe):

    """
    This function takes a pandas DataFrame where each column represents a gene,
    and each row represents a different sample. It returns a DataFrame containing
    the pairwise Pearson correlations between genes.
    """
    # Compute pairwise correlations
    corr = dataframe.corr(method='pearson')
    return corr

def ex_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing value: {val}\nError: {e}")
        return None

def results_conversion(enet1_pred):
    # sample names from one dictionary
    sample_dict = None
    for val in enet1_pred['samples_pred']:
        sample_dict = ex_literal_eval(val)
        if sample_dict is not None:
            break

    if sample_dict is None:
        raise ValueError("No valid dictionaries found in samples_pred")

    sample_names = list(sample_dict.keys())
    transformed_df = pd.DataFrame(index=enet1_pred.index, columns=sample_names)

    for prot in enet1_pred.index:
        converted_dict = ex_literal_eval(enet1_pred.loc[prot, 'samples_pred'])
        if converted_dict is not None:
            for sample in sample_names:
                transformed_df.loc[prot, sample] = converted_dict[sample][1]

    return transformed_df