###create linear model class, one for each gene/protein pair and one that combines all genes for each protein
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
#import cudf
#from cuml.linear_model import ElasticNetCV as cuElasticNetCV
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, SelectKBest


def multilinear_table(transcriptomics, proteomics, protein_name):
    protein_data = proteomics[[protein_name]]
    protein_data.columns = [protein_name + '_prot']
    table = pd.concat([protein_data.reset_index(drop=True), transcriptomics.reset_index(drop=True)], axis=1)
    return table
def feature_selection(table, num_features, protein):
    y = table[protein]
    X = table.drop(protein, axis =1)
    # ANOVA F-test Feature Selection
    selector = SelectKBest(score_func=f_regression, k=num_features) 
    X_new = selector.fit_transform(X, y)

    # Get the selected feature names
    mask = selector.get_support()
    selected_features = X.columns[mask]
    print("Selected Features:", selected_features)
    return table[[protein] + list(selected_features)]
def pca_95(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA().fit(X_scaled)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Determine the number of components that explain at least 95% of the variance
    n_components_95 = np.where(cumulative_variance_ratio >= 0.95)[0][0] + 1

    print(f"Number of components explaining 95% variance: {n_components_95}")
    return n_components_95
def pca_90(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA().fit(X_scaled)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Determine the number of components that explain at least 90% of the variance
    n_components_90 = np.where(cumulative_variance_ratio >= 0.90)[0][0] + 1

    print(f"Number of components explaining 90% variance: {n_components_90}")
    return n_components_90

def plot_cumulative_variance_explained(data, n_components=100):
    """
    Plots the cumulative variance explained by PCA components.
    
    Parameters:
    data (array-like): The input data for PCA.
    n_components (int): Number of principal components to consider.
    
    Returns:
    None
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(data)
    explained_variance = pca.explained_variance_ratio_
    
    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(explained_variance) * 100  # Convert to percentage
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-')
    plt.axhline(y=90, color='r', linestyle='--', label='90% Variance')
    plt.axhline(y=95, color='g', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('')
    plt.title('')
    plt.legend()
    plt.grid(True)
    plt.show()

def pipeline_model(transcriptomics, proteomics, model, num_features, protein, indexes,tolerance = 0.05, k = 10):

    table = multilinear_table(transcriptomics = transcriptomics, proteomics= proteomics, protein_name=protein)
    table.index = indexes
    leave_one_out = table.iloc[0:1 , :]
    target = table.iloc[0, 0]
    table = table[1:] #leave out the table the first row BEFORE feat selection
    table =  feature_selection(table=table, num_features=num_features, protein = protein + '_prot')  
    leave_one_out = leave_one_out[table.columns]#leave_one_out has the same columns as the table after feature selection  
    leave_one_out = leave_one_out.iloc[:, 1:]

    if model == 'ElasticNet': 
         model = ElastNetModel(table = table, left_out = leave_one_out, target = target,  max_iter = 10000, tolerance = tolerance, k = k)
    elif model == 'Multilinear':
        model = MultilinearModel(table = table, left_out = leave_one_out, target = target, tolerance = tolerance, k= k )
    elif model == 'LassoRegression':
        model = LassoRegression(table = table, left_out = leave_one_out, target = target, tolerance = tolerance)
    elif model == 'RidgeRegression':
        model = RidgeRegression(table = table, left_out = leave_one_out, target = target, tolerance = tolerance)
    result_df = pd.DataFrame([model.predict_and_evaluate()])

    return result_df

class ElastNetModel:
    def __init__(self, table, left_out, target, alphas=None, l1_ratios=None, max_iter=10000, tolerance=0.01, k=10):
        """
        Initializes the model using provided data.
        :param table: DataFrame containing both the target and features.
        :param alphas: Range of alpha values for ElasticNetCV.
        :param l1_ratios: Range of l1_ratio values for ElasticNetCV.
        :param max_iter: Maximum iterations for the convergence of the ElasticNetCV.
        """
        self.prot_name = table.columns[0]
        self.prot_train = table.iloc[:, 0]  # Target for training
        self.transcr_train = table.iloc[:, 1:]  # Features for training
        self.prot_test = float(target)  # Ensure target for testing is float
        self.transcr_test = left_out  # Features for testing - retain as DataFrame
        self.tolerance = tolerance
        self.k = k
        self.alphas = alphas if alphas is not None else np.logspace(-4, 0, k)
        self.l1_ratios = l1_ratios if l1_ratios is not None else np.linspace(0.1, 0.9, 5)
        self.max_iter = max_iter

    def predict_and_evaluate(self):
        """
        Fit the model and predict the held-out sample.
        :return: Prediction, actual value, and error metrics.
        """
        # Manually implement cross-validation
        kf = KFold(n_splits=self.k)
        mse_list = []
        r2_list = []
        samples_pred_dict = {}

        for train_index, test_index in kf.split(self.transcr_train):
            X_train, X_test = self.transcr_train.iloc[train_index], self.transcr_train.iloc[test_index]
            y_train, y_test = self.prot_train.iloc[train_index], self.prot_train.iloc[test_index]
            
            model = ElasticNetCV(alphas=self.alphas, l1_ratio=self.l1_ratios, max_iter=self.max_iter, n_jobs=-1)
            model.fit(X_train, y_train)
            
            fold_pred = model.predict(X_test)
            fold_mse = mean_squared_error(y_test, fold_pred)
            fold_r2 = r2_score(y_test, fold_pred)

            sample_names = self.transcr_train.index[test_index].tolist()
            for name, pred in zip(sample_names, fold_pred):
                pred_error = abs(pred - y_test[sample_names.index(name)])
                samples_pred_dict[name] = (pred, pred_error)
            
            mse_list.append(fold_mse)
            r2_list.append(fold_r2)

        # Predict for the test sample
        prediction = model.predict(self.transcr_test)[0]
        actual = self.prot_test
        error = abs(prediction - actual)

        # results
        results = {
            "protein": self.prot_name,  # The name of the protein being evaluated.
            "test_pred": prediction,  # The prediction made by the model for the held-out test sample.
            "test_actual": actual,  # The actual target value for the held-out test sample.
            "test_loss": error,  # The absolute error between the predicted value and the actual value for the held-out test sample.
            "cv_mse_r2": {
                "mse": mse_list,  # List of MSE values for each fold.
                "r2": r2_list,  # List of R2 values for each fold.
            },
            "samples_pred": samples_pred_dict,  # A dictionary with sample names as keys and their predictions as values (tuple of prediction and error).
        }

        print("Best alpha:", model.alpha_)
        print("Best l1 ratio:", model.l1_ratio_)
        print(f'Protein: {self.prot_name}, Error: {error}, Prediction: {prediction}, Actual: {actual}')

        # Check if error is within a certain threshold and print coefficients
        if error < self.tolerance * actual:
            coefficients = dict(zip(self.transcr_train.columns, model.coef_))
            results['coefficients'] = coefficients

        return results
class MultilinearModel:
    def __init__(self, table, left_out, target, max_iter=10000, tolerance=0.01, k=5):
        """
        Initializes the model using provided data.
        :param table: DataFrame containing both the target and features.
        :param left_out: DataFrame containing features for the test sample.
        :param target: Single value, target for the test sample.
        :param max_iter: Maximum iterations for the convergence (placeholder, not used in LinearRegression).
        :param tolerance: Tolerance for considering printing the coefficients.
        :param n_splits: Number of folds for cross-validation.
        """
        self.prot_name = table.columns[0]
        self.prot_train = table.iloc[1:, 0]  # Target for training from second row onwards
        self.transcr_train = table.iloc[1:, 1:]  # Features for training
        self.prot_test = target  # Target for testing (first entry)
        self.transcr_test = left_out  # Features for testing (first row)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_splits = k

    def predict_and_evaluate(self):
        """
        Fit the model using cross-validation and predict the held-out sample.
        :return: Prediction, actual value, and error metrics.
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        model = LinearRegression()
        
        # Perform cross-validati
        model.fit(self.transcr_train, self.prot_train)
        prediction = model.predict(self.transcr_test)[0]
        actual = self.prot_test
        error = abs(prediction - actual)

        # Organize results
        results = {
            "protein": self.prot_name,
            "prediction": prediction,
            "actual": actual,
            "loss": error,
        }
        print(f'Protein: {self.prot_name}, Error: {error}, prediction: {prediction}, actual: {actual}')
        
        # Check if error is within a certain threshold and print coefficients
        if error < self.tolerance * actual:
            coefficients = dict(zip(self.transcr_train.columns, model.coef_))
            results['coefficients'] = coefficients

        return results

class LassoRegression():
    def __init__(self, table, left_out, target, max_iter=10000, tolerance=0.01):

        self.prot_name = table.columns[0]
        self.prot_train = table.iloc[1:, 0]  # Target for training from second row onwards
        self.transcr_train = table.iloc[1:, 1:]  # Features for training
        self.prot_test = target  # Target for testing (first entry)
        self.transcr_test = left_out  # Features for testing (first row)
        self.max_iter = max_iter
        self.tolerance = tolerance  # Features for testing (first row) - retain as DataFrame
    def predict_and_evaluate(self):
        """
        Fit the model and predict the held-out sample.
        :return: Prediction, actual value, and error metrics.
        """
        model = LassoCV(cv = 10, random_state = 42)
        model.fit(self.transcr_train, self.prot_train)
        
        prediction = model.predict(self.transcr_test)[0]
        error = abs(prediction - self.prot_test)
        results = {
        "protein": self.prot_name,
        "prediction": prediction,
        "actual": self.prot_test,
        "loss": error
        }
        print(f'Protein: {self.prot_name}, error: {error}, prediction: {prediction}, actual: {self.prot_test}')
        if error < self.tolerance * abs(self.prot_test):
            coefficients = dict(zip(self.transcr_train.columns, model.coef_))
            results['coefficients'] = coefficients

        return results
    
class RidgeRegression():
    def __init__(self, table, left_out, target, max_iter=10000, tolerance=0.01):
        """
        Initializes the model using provided data.
        :param table: DataFrame containing both the target and features.
        :param left_out: DataFrame containing features for the test sample.
        :param target: Single value, target for the test sample.
        :param max_iter: Maximum iterations for the convergence of RidgeCV.
        :param tolerance: Tolerance for considering printing the coefficients.
        """
        self.prot_name = table.columns[0]
        self.prot_train = table.iloc[1:, 0]  # Target for training from second row onwards
        self.transcr_train = table.iloc[1:, 1:]  # Features for training
        self.prot_test = target  # Target for testing (first entry)
        self.transcr_test = left_out  # Features for testing (first row)
        self.max_iter = max_iter
        self.tolerance = tolerance

    def predict_and_evaluate(self):
        """
        Fit the model and predict the held-out sample.
        :return: Prediction, actual value, and error metrics.
        """
        model = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=10)
        model.fit(self.transcr_train, self.prot_train)
        
        prediction = model.predict(self.transcr_test)[0]
        error = abs(prediction - self.prot_test)  # Use absolute error for consistency
        
        results = {
            "protein": self.prot_name,
            "prediction": prediction,
            "actual": self.prot_test,
            "loss": error
        }
        print(f'Protein: {self.prot_name}, Error: {error}, Prediction: {prediction}, Actual: {self.prot_test}')
        
        if error < self.tolerance * abs(self.prot_test):  # Check against absolute value of actual
            coef_df = pd.DataFrame(model.coef_, index=self.transcr_train.columns, columns=['Coefficients'])
            results['coefficients'] = coef_df.to_dict()

        return results

