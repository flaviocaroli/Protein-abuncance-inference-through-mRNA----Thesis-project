# combined_model.py

import torch
import torch.nn as nn
from model import CarNet, Autoencoder, CombinedModel
from utils import load_data

def load_combined_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, _ , _, _ = load_data()
    input_dim = X_train.shape[1]  # Original input dimension before autoencoder
    latent_dim = 128

    # Ensure this matches the output_shape used during training
    output_dim = 5108  # This should match the output_shape from training

    # Load models
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    carnet = CarNet(input_shape=latent_dim, output_shape=output_dim, dropout_rate=0.2).to(device)
    combined_model = CombinedModel(autoencoder, carnet).to(device)

    # Load pretrained weights
    autoencoder.encoder.load_state_dict(torch.load('autoencoder_encoder.pth'))
    carnet.load_state_dict(torch.load('carnet_best.pth'))

    # Freeze the encoder parameters if necessary
    for param in autoencoder.encoder.parameters():
        param.requires_grad = False

    print("Combined model loaded with pretrained weights.")
    return combined_model

# Example usage
if __name__ == "__main__":
    model = load_combined_model()
    print(model)
