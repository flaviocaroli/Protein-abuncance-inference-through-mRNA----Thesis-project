# main.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from model import Autoencoder, CarNet, CombinedModel
from utils import load_data

# Function to train the combined model
def train_model():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    X_train, X_test, y_train, y_test = load_data()
    
    # Prepare data loaders
    batch_size = 32  # Example batch size
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    input_dim = X_train.shape[1]
    latent_dim = 128
    output_dim = y_train.shape[1]

    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    carnet = CarNet(input_shape=latent_dim, output_shape=output_dim, dropout_rate=0.2).to(device)
    model = CombinedModel(autoencoder, carnet).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    num_epochs = 100
    best_loss = np.inf

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Move data to GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}')

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # Move data to GPU
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        print(f'Test Loss: {test_loss}')
        
        # Model checkpointing
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'protein_predictor_best.pth')
    
    # Save the encoded data
    save_encoded_data(autoencoder, X_train, X_test, device)

    return best_loss

def save_encoded_data(autoencoder, X_train, X_test, device):
    autoencoder.eval()
    
    # Move input data to the same device as the autoencoder
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Encode the training and test data
    X_train_encoded = autoencoder.encode(X_train_tensor).detach().cpu().numpy()
    X_test_encoded = autoencoder.encode(X_test_tensor).detach().cpu().numpy()
    
    # Save the encoded data to CSV files
    pd.DataFrame(X_train_encoded).to_csv('X_train_encoded.csv', index=False)
    pd.DataFrame(X_test_encoded).to_csv('X_test_encoded.csv', index=False)
    print("Encoded data saved to 'X_train_encoded.csv' and 'X_test_encoded.csv'")

# Example usage
if __name__ == "__main__":
    best_loss = train_model()
    print(f'Best loss: {best_loss}')
