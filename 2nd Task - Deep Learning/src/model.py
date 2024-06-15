import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),  # speed up training and stabilize
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512), 
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

class CarNet(nn.Module):
    def __init__(self, input_shape, output_shape, dropout_rate=0.2, hidden_size=50):
        super(CarNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.bn4 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4)
        
        self.conv_output_size = self.calculate_conv_output_size(input_shape)
        self.fc1 = nn.Linear(self.conv_output_size * hidden_size * 2, 128)
        
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, output_shape)
        
    def calculate_conv_output_size(self, input_size):
        conv_output_size = input_size
        conv_output_size = conv_output_size - 2  # After first conv
        conv_output_size = conv_output_size - 2  # After second conv
        conv_output_size = conv_output_size - 2  # After third conv
        return conv_output_size
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm(x)
        x = self.bn4(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout2(x)
        
        x, _ = self.attention(x, x, x)
        
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, autoencoder, carnet):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder
        self.carnet = carnet

    def forward(self, x):
        x = self.autoencoder.encode(x)
        x = x.unsqueeze(1)  # Adding a channel dimension for Conv1d
        x = self.carnet(x)
        return x
