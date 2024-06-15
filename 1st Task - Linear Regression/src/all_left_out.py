import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error

def multilinear_table(transcriptomics, proteomics, protein_name):
    protein_data = proteomics[[protein_name]]
    protein_data.columns = [protein_name + '_prot']
    table = pd.concat([protein_data.reset_index(drop=True), transcriptomics.reset_index(drop=True)], axis=1)
    return table

def feature_selection(table, num_features, protein):
    y = table[protein]
    X = table.drop(protein, axis=1)
    selector = SelectKBest(score_func=f_regression, k=num_features)
    X_new = selector.fit_transform(X, y)

    mask = selector.get_support()
    selected_features = X.columns[mask]
    return table[[protein] + list(selected_features)], selected_features

class ParallelElasticNet:
    def __init__(self, table, left_out, target, max_iter=10000, alpha=0.02, l1_ratio=0.1):
        self.prot_name = table.columns[0]
        self.prot_train = table.iloc[:, 0].values
        self.transcr_train = table.iloc[:, 1:].values
        self.prot_test = target
        self.transcr_test = left_out.values
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def predict_and_evaluate(self):
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
        model.fit(self.transcr_train, self.prot_train)
        prediction = model.predict(self.transcr_test)[0]
        actual = self.prot_test
        error = abs(prediction - actual)

        results = {
            "protein": self.prot_name,
            "loss": error,
        }
        return results

def pipeline_AllLeftOut(transcriptomics, proteomics, num_features, protein, indexes, tissue_samples):
    print(f"Processing protein: {protein}")
    table = multilinear_table(transcriptomics=transcriptomics, proteomics=proteomics, protein_name=protein)
    table.index = indexes
    errors = {}
    selected_features = set()

    for tissue, samples in tissue_samples.items():
        for sample_name in samples:
            print(f"  Processing sample: {sample_name}")
            leave_out = table.loc[[sample_name]]
            target = table.loc[sample_name, table.columns[0]]
            training_data = table.drop(sample_name)
            training_data, current_features = feature_selection(table=training_data, num_features=num_features, protein=protein + '_prot')
            selected_features = selected_features.union(current_features)
            leave_out = leave_out[training_data.columns]
            leave_out = leave_out.iloc[:, 1:]

            model = ParallelElasticNet(table=training_data, left_out=leave_out, target=target, max_iter=10000)
            result = model.predict_and_evaluate()
            errors[sample_name] = result['loss']

    union_features_str = ', '.join(selected_features)
    result_row = pd.Series(errors, name=protein)
    result_row['features'] = union_features_str
    print(f"Completed processing protein: {protein}")
    return result_row

def count_samples_by_tissue(transcriptomics):
    counts_dict = {}
    for sample in transcriptomics.index:
        if '_' not in sample:
            continue
        name = sample.split('_', 1)[1]
        counts_dict[name] = counts_dict.get(name, 0) + 1
    return counts_dict

def select_samples_by_tissue1(counts_dict, transcriptomics):
    tissue_samples = {}
    for tissue, count in counts_dict.items():
        samples = [sample for sample in transcriptomics.index if tissue in sample]
        tissue_samples[tissue] = samples
    return tissue_samples


