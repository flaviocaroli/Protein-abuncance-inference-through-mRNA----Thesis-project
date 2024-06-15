import datetime
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

################# Standardized Pipeline Functions #################

def get_common_proteins(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Index:
    """
    Identifies and returns common proteins (indices) between two DataFrames.

    Parameters:
    - df1 (pd.DataFrame): The first DataFrame with proteins as index.
    - df2 (pd.DataFrame): The second DataFrame with proteins as index.

    Returns:
    - pd.Index: The index of common proteins between the two DataFrames.
    """
    return df1.index.intersection(df2.index)

def get_common_samples(df1: pd.DataFrame, df2: pd.DataFrame) -> np.ndarray:
    """
    Identifies and returns common samples (columns) between two DataFrames.

    Parameters:
    - df1 (pd.DataFrame): The first DataFrame with samples as columns.
    - df2 (pd.DataFrame): The second DataFrame with samples as columns.

    Returns:
    - np.ndarray: An array of common sample names between the two DataFrames.
    """
    return np.intersect1d(df1.columns, df2.columns)

def match_proteins_samples(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """
    Filters both DataFrames to only include common proteins (index) and common samples (columns),
    then prints the counts of these common elements.

    Parameters:
    - df1 (pd.DataFrame): The first DataFrame.
    - df2 (pd.DataFrame): The second DataFrame.

    Returns:
    - tuple: A pair of DataFrames filtered to include only the common proteins and samples.
    """
    common_proteins = get_common_proteins(df1, df2)
    common_samples = get_common_samples(df1, df2)
    print("Number of common proteins:", len(common_proteins))
    print("Number of common samples:", len(common_samples))
    
    return df1.loc[common_proteins, common_samples], df2.loc[common_proteins, common_samples]

def correlate_genewise(df1: pd.DataFrame, df2: pd.DataFrame, cname: str, method: str = 'spearman') -> pd.DataFrame:
    """
    Computes correlation between the corresponding elements of two DataFrames, row-wise.

    Parameters:
    - df1 (pd.DataFrame): The first DataFrame.
    - df2 (pd.DataFrame): The second DataFrame.
    - cname (str): Column name for the resulting DataFrame containing correlations.
    - method (str): The method of correlation (e.g., 'pearson', 'spearman').

    Returns:
    - pd.DataFrame: A DataFrame containing the correlations with a specified column name.
    """
    correlation = df1.corrwith(df2, axis=1, method=method)
    print("Median", method.title(), "Correlation:", round(correlation.median(), 4))
    return correlation.to_frame(cname)

def dropna(dataframe: pd.DataFrame, non_null_threshold: float = 0.8, replace_zero: bool = False) -> pd.DataFrame:
    """
    Drops rows from a DataFrame based on a threshold of non-null values, optionally replacing zeros with NaN.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to process.
    - non_null_threshold (float): The minimum proportion of non-null values required to keep a row.
    - replace_zero (bool): Whether to replace zero values with NaN before dropping rows.

    Returns:
    - pd.DataFrame: The processed DataFrame with rows dropped based on the given threshold.
    """
    if replace_zero:
        dataframe = dataframe.replace(0, np.nan)
    non_null_columns = len(dataframe.columns) * non_null_threshold
    return dataframe.dropna(thresh=non_null_columns)

def process(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the DataFrame by replacing zeros with NaN (if no NaNs are present),
    dropping rows with a high proportion of NaNs, computing the mean for protein isoforms,
    and removing specific unwanted index entries.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to be processed.

    Returns:
    - pd.DataFrame: The processed DataFrame.
    """
    replace_zero = dataframe.isnull().sum().sum() == 0
    dataframe_processed = dropna(dataframe, replace_zero=replace_zero)
    dataframe_processed = dataframe_processed.groupby(dataframe_processed.index).mean()
    # Removing rows where the index is a datetime object or contains a colon
    dataframe_processed = dataframe_processed[~dataframe_processed.index.map(lambda x: isinstance(x, datetime.datetime) or ':' in str(x))]
    print("Dimensions:", dataframe_processed.shape)
    return dataframe_processed
