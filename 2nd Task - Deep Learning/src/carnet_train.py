import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import CarNet
import random

def calculate_accuracy(y_true, y_pred, threshold=0.1):
    correct = torch.abs(y_true - y_pred) < threshold
    accuracy = correct.sum().item() / torch.numel(correct)
    return accuracy

def train_carnet(params, save_model=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    latent_train = np.load('latent_train.npy')
    latent_test = np.load('latent_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    batch_size = params['batch_size']
    
    train_dataset = TensorDataset(torch.tensor(latent_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(latent_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_shape = latent_train.shape[1]
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

    # Generate and save predictions
    carnet.eval()
    train_predictions = []
    test_predictions = []
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).unsqueeze(1)
            outputs = carnet(X_batch)
            train_predictions.append(outputs.cpu().numpy())

        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device).unsqueeze(1)
            outputs = carnet(X_batch)
            test_predictions.append(outputs.cpu().numpy())

    train_predictions = np.vstack(train_predictions)
    test_predictions = np.vstack(test_predictions)

    # Create dataframes for train and test predictions
    train_pred_df = pd.DataFrame(train_predictions, columns=[f'pred_{i}' for i in range(train_predictions.shape[1])])
    test_pred_df = pd.DataFrame(test_predictions, columns=[f'pred_{i}' for i in range(test_predictions.shape[1])])
    
    train_actual_df = pd.DataFrame(y_train, columns=[f'actual_{i}' for i in range(y_train.shape[1])])
    test_actual_df = pd.DataFrame(y_test, columns=[f'actual_{i}' for i in range(y_test.shape[1])])
    
    train_df = pd.concat([train_pred_df, train_actual_df], axis=1)
    test_df = pd.concat([test_pred_df, test_actual_df], axis=1)
    
    train_df.to_csv('train_predictions.csv', index=False)
    test_df.to_csv('test_predictions.csv', index=False)

    print("Predictions saved to train_predictions.csv and test_predictions.csv")
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss', color='lightblue')
    plt.plot(range(num_epochs), test_losses, label='Test Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy', color='lightblue')
    plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracies')
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
        'batch_size': [16, 32, 64],
        'weight_decay': [1e-2, 1e-3, 1e-4]
    }
    random_search(param_grid, n_iter=20)
