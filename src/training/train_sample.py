import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.data.process_data import DataProcessor
from src.models.transformer_model import TradingTransformer, TradingConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class TradingDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        
        # Convert to tensor if not already
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.float32)
        
        return sequence, torch.tensor(label, dtype=torch.float32)

def prepare_combined_data(processor):
    """Prepare combined training and testing datasets for Bitcoin and Gala"""
    # Prepare Bitcoin data
    btc_train_images, btc_train_labels, btc_test_images, btc_test_labels = processor.prepare_dataset(
        crypto_name='bitcoin',
        window_size=8,
        test_size=0.2
    )
    
    # Prepare Gala data
    gala_train_images, gala_train_labels, gala_test_images, gala_test_labels = processor.prepare_dataset(
        crypto_name='gala',
        window_size=8,
        test_size=0.2
    )
    
    # Convert numpy arrays to tensors
    btc_train_tensors = torch.from_numpy(btc_train_images).float()
    btc_train_labels = torch.from_numpy(btc_train_labels).float()
    btc_test_tensors = torch.from_numpy(btc_test_images).float()
    btc_test_labels = torch.from_numpy(btc_test_labels).float()
    
    gala_train_tensors = torch.from_numpy(gala_train_images).float()
    gala_train_labels = torch.from_numpy(gala_train_labels).float()
    gala_test_tensors = torch.from_numpy(gala_test_images).float()
    gala_test_labels = torch.from_numpy(gala_test_labels).float()
    
    # Combine datasets
    train_tensors = torch.cat([btc_train_tensors, gala_train_tensors], dim=0)
    train_labels = torch.cat([btc_train_labels, gala_train_labels], dim=0)
    test_tensors = torch.cat([btc_test_tensors, gala_test_tensors], dim=0)
    test_labels = torch.cat([btc_test_labels, gala_test_labels], dim=0)
    
    # Create datasets
    train_dataset = TradingDataset(train_tensors, train_labels)
    test_dataset = TradingDataset(test_tensors, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

def train_model(train_loader, test_loader):
    # Initialize model with configuration
    config = TradingConfig(
        input_dim=6,  # Open, High, Low, Close, Volume, Marketcap/Quote asset volume
        hidden_dim=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    )
    model = TradingTransformer(config)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Ensure labels have the same shape as outputs
            labels = labels.unsqueeze(1)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
        val_loss /= len(test_loader)
        
        print(f'Epoch [{epoch+1}/50], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with validation loss: {best_val_loss:.4f}')
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on both Bitcoin and Gala separately
    evaluate_model_performance(model, test_loader, 'bitcoin', device)
    evaluate_model_performance(model, test_loader, 'gala', device)
    
    return model

def evaluate_model_performance(model, test_loader, crypto_name, device):
    """
    Evaluate model performance on test data and generate comprehensive metrics.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        crypto_name: Name of the cryptocurrency (for reporting)
        device: Device to run the evaluation on
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print metrics
    print(f"\n{crypto_name} Performance Metrics:")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")
    print(f"- F1 Score: {f1:.4f}")
    print(f"- AUC: {auc:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("Predicted\tActual")
    print(f"{cm[0][0]}\tTP\t{cm[0][1]}\tFP")
    print(f"{cm[1][0]}\tFN\t{cm[1][1]}\tTN")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Down', 'Up'],
               yticklabels=['Down', 'Up'])
    plt.title(f'{crypto_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'evaluation_{crypto_name.lower()}_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{crypto_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'evaluation_{crypto_name.lower()}_roc_curve.png')
    plt.close()
    
    # Return metrics for further analysis
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

def calculate_profits(predictions, actual_prices):
    """Calculate estimated profit based on predictions"""
    # This is a simple profit calculation based on predictions
    # In a real trading system, you'd want to consider:
    # - Transaction costs
    # - Position sizing
    # - Risk management
    # - Slippage
    
    profits = []
    for pred, actual in zip(predictions, actual_prices):
        if pred == 1:  # Predicted increase
            profit = (actual[1] - actual[0]) / actual[0] * 100  # % change
        else:  # Predicted decrease
            profit = (actual[0] - actual[1]) / actual[0] * 100  # % change
        profits.append(profit)
    
    return sum(profits) / len(profits) if profits else 0.0

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data processor with data directory
    processor = DataProcessor(data_dir='data/processed')
    
    # Prepare combined data
    train_loader, test_loader = prepare_combined_data(processor)
    
    # Train model
    trained_model = train_model(train_loader, test_loader)
    
    # Save model
    torch.save(trained_model.state_dict(), 'trained_model.pth')
    print('Model saved!')
    num_epochs = 50
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate accuracy
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved!')
    
    print('Training complete!')
    
    # Evaluate on both Bitcoin and Gala separately
    evaluate_on_crypto(model, 'bitcoin', test_loader)
    evaluate_on_crypto(model, 'gala', test_loader)

if __name__ == "__main__":
    main()
