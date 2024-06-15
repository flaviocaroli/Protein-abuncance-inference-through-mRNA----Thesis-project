import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
from sklearn.preprocessing import MinMaxScaler


def dropna(dataframe, replace_zero=False):
    if replace_zero:
        return dataframe.fillna(0)
    else:
        return dataframe.dropna()

def process(dataframe: pd.DataFrame) -> pd.DataFrame:
    replace_zero = dataframe.isnull().sum().sum() == 0
    dataframe_processed = dropna(dataframe, replace_zero=replace_zero)
    dataframe_processed = dataframe_processed.groupby(dataframe_processed.index).mean()
    # Removing rows where the index is a datetime object or contains a colon
    dataframe_processed = dataframe_processed[~dataframe_processed.index.map(lambda x: isinstance(x, datetime.datetime) or ':' in str(x))]
    print("Dimensions:", dataframe_processed.shape)
    return dataframe_processed

def load_data():

    #Load 
    #mutations = pd.read_csv('../Data/CCLE_mutations.csv')
    expression  = pd.read_csv('../Data/CCLE_expression.csv', index_col= 0)
    #cnv = pd.read_csv('../Data/CCLE_gene_cn.csv', index_col= 0)
    prot_normalized = pd.read_csv('../Data/protein_quant_current_normalized.csv', index_col= 0)
    sample_info = pd.read_csv('../Data/sample_info.csv')

    id_to_name_map = dict(zip(sample_info['DepMap_ID'], sample_info['CCLE_Name']))

    '''   #### Mutations
    mutations['mutation'] = 1
    binary_matrix = mutations.pivot_table(
        index='DepMap_ID',
        columns='Hugo_Symbol',
        values='mutation',
        fill_value=0
    )

    mutations['count'] = 1
    mutation_types = mutations.pivot_table(
        index='DepMap_ID',
        columns='Variant_Classification',
        values='count',
        fill_value=0
    )
    mutation_types_columns = ['_'.join(col).strip() for col in mutation_types.columns.values]

    mutation_features = pd.concat([binary_matrix, mutation_types], axis=1)
    mutation_features = mutation_features.rename(index = id_to_name_map)
    mutation_processed = process(mutation_features)'''

    #### Copy NumberEncoding
   # cnv = cnv.rename(index = id_to_name_map)
   # cnv_processed = process(cnv) 

    #### Expression
    expression = expression.rename(index = id_to_name_map)
    expression_processed = process(expression)

    #### Proteomics
    prot_normalized.set_index('Gene_Symbol', inplace=True)
    proteomics = prot_normalized.loc[:,  prot_normalized.columns.str.contains('_TenPx')]
    proteomics.drop(columns=['SW948_LARGE_INTESTINE_TenPx11', 'CAL120_BREAST_TenPx02', 'HCT15_LARGE_INTESTINE_TenPx30'], 
                        inplace=True)
    proteomics = proteomics.rename(columns = lambda x : str(x).split('_TenPx')[0])
    proteomics_processed = process(proteomics)
    proteomics_processed = proteomics_processed.transpose()
    print(proteomics_processed.shape)

    #### Matching Samples
    #common_samples = set(expression_processed.index) & set(mutation_processed.index) & set(cnv_processed.index) & set(proteomics_processed.index)
    common_samples = set(expression_processed.index) & set(proteomics_processed.index)
    common_samples = list(common_samples)   
    print(len(common_samples))

    #cnv_data = cnv_processed.loc[common_samples]
    mrna_data = expression_processed.loc[common_samples]
    #mutation_features = mutation_processed.loc[common_samples]
    proteomics_data = proteomics_processed.loc[common_samples]
    #combined_features = pd.concat([mrna_data, cnv_data, mutation_features], axis=1)
    combined_features = pd.concat([mrna_data], axis=1)
    print(combined_features.shape)

    # MinMax scale 
    scaler = MinMaxScaler()
    X = scaler.fit_transform(combined_features.values)
    y = scaler.fit_transform(proteomics_data.values)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

