# carnet_train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from model import CarNet
import random
from utils import load_data

def calculate_accuracy(y_true, y_pred, threshold=0.1):
    correct = torch.abs(y_true - y_pred) < threshold
    accuracy = correct.sum().item() / torch.numel(correct)
    return accuracy

def train_carnet(params, save_model=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    # Load data directly from load_data()
    X_train, X_test, y_train, y_test = load_data()

    batch_size = params['batch_size']
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]
    carnet = CarNet(input_shape=input_shape, output_shape=output_shape, dropout_rate=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(carnet.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    num_epochs = 100

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    best_loss = float('inf')

    for epoch in range(num_epochs):
        carnet.train()
        train_loss = 0
        train_accuracy = 0
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            X_batch = X_batch.unsqueeze(1)

            outputs = carnet(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_accuracy += calculate_accuracy(y_batch, outputs)
        
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}')

        carnet.eval()
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch = X_batch.unsqueeze(1)
                outputs = carnet(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
                test_accuracy += calculate_accuracy(y_batch, outputs)

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        
        if save_model and test_loss < best_loss:
            best_loss = test_loss
            torch.save(carnet.state_dict(), 'carnet_best.pth')
    
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
    plt.savefig('carnet_metrics.png')
    plt.show()
    
    return train_losses, test_losses

def random_search(param_grid, n_iter=10):
    results = {'params': [], 'train_losses': [], 'test_losses': []}
    best_params = None
    best_loss = float('inf')

    for i in range(n_iter):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"Iteration {i+1}/{n_iter}: Testing with params: {params}")
        train_losses, test_losses = train_carnet(params, save_model=False)
        results['params'].append(params)
        results['train_losses'].append(train_losses)
        results['test_losses'].append(test_losses)
        if min(test_losses) < best_loss:
            best_loss = min(test_losses)
            best_params = params

    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_loss}")
    train_carnet(best_params, save_model=True)
    
    

if __name__ == "__main__":
    param_grid = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'batch_size': [16],
        'weight_decay': [1e-2, 1e-3, 1e-4]
    }
    random_search(param_grid, n_iter=20)
