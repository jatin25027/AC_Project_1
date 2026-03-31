"""
Train and evaluate neural distinguishers.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import DEVICE, save_results
from input_representations.data_processing import prepare_representation
from ml_models.models import MLP, CNN, SiameseNet, MINE, mine_loss

def calculate_accuracy(y_pred, y_true):
    predicted = (torch.sigmoid(y_pred) > 0.5).float()
    correct = (predicted == y_true.unsqueeze(1)).sum().item()
    return correct / y_true.size(0)

def train_model(model, train_loader, val_loader, model_type='MLP', epochs=20, lr=0.001):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if model_type != 'MINE':
        # Standard binary classification
        criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            if model_type == 'MINE':
                # For MINE, x=C, y=C' split from data
                # Assuming data is [B, 2, C]
                x = data[:, 0, :]
                y_ = data[:, 1, :]
                
                # Only use Cipher pairs (label==1) to maximize MI
                cipher_mask = (labels == 1)
                if cipher_mask.sum() > 1:
                    loss = mine_loss(model, x[cipher_mask], y_[cipher_mask])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
            else:
                outputs = model(data)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += calculate_accuracy(outputs, labels)
                
        # Validation
        val_acc = evaluate(model, val_loader, model_type)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    if best_model_state:
        model.load_state_dict(best_model_state)
    return best_acc

def evaluate(model, loader, model_type):
    model.eval()
    acc = 0.0
    total = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            if model_type == 'MINE':
                x = data[:, 0, :]
                y_ = data[:, 1, :]
                # Distinguisher threshold for MINE based on T(x,y)
                t = model(x, y_).squeeze()
                # Assuming positive t means dependent (cipher)
                pred = (t > 0).float()
                acc += (pred == labels).sum().item()
            else:
                outputs = model(data)
                acc += ((torch.sigmoid(outputs) > 0.5).float() == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
    return acc / total if total > 0 else 0.5
