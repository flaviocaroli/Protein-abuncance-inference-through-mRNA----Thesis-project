import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime
from torch import nn
from model import Autoencoder, CarNet  # Assuming these classes are defined in model.py

def load_data2():
    expression = pd.read_csv('../Data/CCLE_expression.csv', index_col=0)
    prot_normalized = pd.read_csv('../Data/protein_quant_current_normalized.csv', index_col=0)
    sample_info = pd.read_csv('../Data/sample_info.csv')

    id_to_name_map = dict(zip(sample_info['DepMap_ID'], sample_info['CCLE_Name']))

    expression = expression.rename(index=id_to_name_map)
    expression_processed = expression.groupby(expression.index).mean()
    expression_processed = expression_processed[~expression_processed.index.map(lambda x: isinstance(x, datetime.datetime) or ':' in str(x))]

    prot_normalized.set_index('Gene_Symbol', inplace=True)
    proteomics = prot_normalized.loc[:, prot_normalized.columns.str.contains('_TenPx')]
    proteomics = proteomics.rename(columns=lambda x: str(x).split('_TenPx')[0])
    
    proteomics.drop(columns=['SW948_LARGE_INTESTINE_TenPx11', 'CAL120_BREAST_TenPx02', 'HCT15_LARGE_INTESTINE_TenPx30'], inplace=True, errors='ignore')
    proteomics_processed = proteomics.groupby(proteomics.columns, axis=1).mean()
    proteomics_processed = proteomics_processed.transpose()

    common_samples = set(expression_processed.index) & set(proteomics_processed.index)
    common_samples = list(common_samples)

    mrna_data = expression_processed.loc[common_samples]
    proteomics_data = proteomics_processed.loc[common_samples]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(mrna_data.values)
    y = scaler.fit_transform(proteomics_data.values)
    
    return X, y, mrna_data.index, proteomics_data.columns

class CombinedModel(nn.Module):
    def __init__(self, autoencoder, carnet, latent_dim, input_shape):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder.encoder  # Only use the encoder part of the autoencoder
        self.fc_adjust = nn.Linear(latent_dim, input_shape)  # Adjust dimensions
        self.carnet = carnet

    def forward(self, x):
        x = self.autoencoder(x)
        x = self.fc_adjust(x)
        x = x.unsqueeze(1)  # Adding a channel dimension for Conv1d
        x = self.carnet(x)
        return x

def main():
    X, y, sample_index, proteomics_columns = load_data2()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Loadù trained models
    autoencoder = Autoencoder(input_dim=X.shape[1], latent_dim=256)
    autoencoder.encoder.load_state_dict(torch.load('autoencoder_encoder.pth', map_location=device))
    autoencoder.to(device)

    # Ensure you use the same input_shape and output_shape as during training
    input_shape = 1250  # Adjust to match the saved model's input to fc1
    output_shape = y.shape[1]  # Should match the output dimension

    carnet = CarNet(input_shape=input_shape, output_shape=output_shape, dropout_rate=0.2)
    carnet.load_state_dict(torch.load('carnet_best.pth', map_location=device))
    carnet.to(device)

    combined_model = CombinedModel(autoencoder, carnet, latent_dim=256, input_shape=input_shape)
    combined_model.to(device)
    combined_model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # save predictions
    with torch.no_grad():
        predictions = combined_model(X_tensor).cpu().numpy()s
    predictions_df = pd.DataFrame(predictions, index=sample_index, columns=proteomics_columns)
    predictions_df.to_csv('predictions.csv')
    print(predictions_df.head())

if __name__ == "__main__":
    main()
