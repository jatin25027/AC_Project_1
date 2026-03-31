"""
Machine Learning models for neural distinguishers.
1. MLP
2. CNN (1D ResNet like)
3. Siamese Network
4. MINE (Mutual Information Neural Estimator)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. Multi-Layer Perceptron (MLP)
# ==============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten arbitrary shapes
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

# ==============================================================================
# 2. Convolutional Neural Network (CNN) - ResNet inspired
# ==============================================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, dim, k=3):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(dim, dim, k, padding=k//2)
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, k, padding=k//2)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class CNN(nn.Module):
    def __init__(self, input_channels, seq_len=64, hidden_dim=64):
        super(CNN, self).__init__()
        self.conv_in = nn.Conv1d(input_channels, hidden_dim, 3, padding=1)
        self.res1 = ResidualBlock1D(hidden_dim)
        self.res2 = ResidualBlock1D(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [B, channels, length] (e.g. rep 1 gives [B, 2, 64])
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, 1, seq_len]
        elif x.dim() == 3:
            # If the current channel count doesn't match the weight but the length does, transpose.
            # This handles [B, L, C] inputs.
            if x.shape[1] != self.conv_in.in_channels and x.shape[2] == self.conv_in.in_channels:
                x = x.transpose(1, 2)
        out = F.relu(self.conv_in(x))
        out = self.res1(out)
        out = self.res2(out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

# ==============================================================================
# 3. Siamese Network
# ==============================================================================
class SiameseNet(nn.Module):
    def __init__(self, branch_dim, hidden_dim=64):
        super(SiameseNet, self).__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        # Expected shape [B, 2, branch_dim]
        # x1 is from ciphertext C, x2 is from C'
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        feat1 = self.branch(x1)
        feat2 = self.branch(x2)
        combined = torch.cat([feat1, feat2], dim=1)
        return self.fc(combined)

# ==============================================================================
# 4. Mutual Information Neural Estimator (MINE)
# ==============================================================================
class MINE(nn.Module):
    """
    Estimates MI between ciphertexts (C) and differences (C').
    For distinguisher, we output scalar discrimination from joint stats.
    """
    def __init__(self, x_dim, y_dim, hidden_size=64):
        super(MINE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, y):
        # x is from joint distribution P(x,y), y is from marginals
        # Output is essentially T(x,y) for MINE objective
        joint = torch.cat((x, y), dim=1)
        return self.net(joint)

def mine_loss(net, x, y):
    """
    Computes MINE loss and mutual information estimate explicitly.
    In the context of distinguisher, we maximize MI on cipher data.
    """
    # shuffle y to get marginals
    y_shuffled = y[torch.randperm(y.shape[0])]
    
    t = net(x, y)
    t_marg = net(x, y_shuffled)
    
    mi = torch.mean(t) - torch.log(torch.mean(torch.exp(t_marg)))
    return -mi  # minimize negative MI

