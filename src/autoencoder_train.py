import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random
from model import Autoencoder
from utils import load_data

def calculate_accuracy(y_true, y_pred, threshold=0.1):
    correct = torch.abs(y_true - y_pred) < threshold
    accuracy = correct.sum().item() / torch.numel(correct)
    return accuracy

def train_autoencoder(params, save_model=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    X_train, X_test, y_train, y_test = load_data()

    batch_size = params['batch_size']
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape[1]
    latent_dim = 256
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    num_epochs = 30
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        autoencoder.train()
        train_loss = 0
        train_accuracy = 0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(X_batch)
            loss = criterion(outputs, X_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += calculate_accuracy(X_batch, outputs)
        
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        autoencoder.eval()
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = autoencoder(X_batch)
                loss = criterion(outputs, X_batch)
                test_loss += loss.item()
                test_accuracy += calculate_accuracy(X_batch, outputs)
        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')
    
    if save_model:
        torch.save(autoencoder.encoder.state_dict(), 'autoencoder_encoder.pth')
        autoencoder.eval()
        with torch.no_grad():
            latent_train = autoencoder.encode(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
            latent_test = autoencoder.encode(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        
        np.save('latent_train.npy', latent_train)
        np.save('latent_test.npy', latent_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)

        print("Autoencoder training complete and latent representations saved.")
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss', color='lightblue')
    plt.plot(range(num_epochs), test_losses, label='Test Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy', color='lightblue')
    plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('autoencoder_metrics.png')
    plt.show()

    return train_losses, test_losses

def random_search(param_grid, n_iter=10):
    results = {'params': [], 'train_losses': [], 'test_losses': []}
    best_params = None
    best_loss = float('inf')

    for i in range(n_iter):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"Iteration {i+1}/{n_iter}: Testing with params: {params}")
        train_losses, test_losses = train_autoencoder(params, save_model=False)
        results['params'].append(params)
        results['train_losses'].append(train_losses)
        results['test_losses'].append(test_losses)
        if min(test_losses) < best_loss:
            best_loss = min(test_losses)
            best_params = params

    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_loss}")
    train_autoencoder(best_params, save_model=True)
    
  
if __name__ == "__main__":
    param_grid = {
        'learning_rate': [0.001, 0.0001],
        'batch_size': [8, 16, 32],
        'weight_decay': [ 1e-3, 1e-4]
    }
    random_search(param_grid, n_iter=20)
