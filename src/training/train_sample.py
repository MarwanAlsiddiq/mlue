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

# Initialize config
config = TradingConfig()

class TradingConfig:
    def __init__(self):
        self.input_dim = 10  # open, high, low, close, volume, marketcap, rsi, macd, macd_signal, macd_hist
        self.hidden_dim = 512
        self.num_layers = 6
        self.num_heads = 8
        self.dropout = 0.3
        self.learning_rate = 0.0001
        self.weight_decay = 0.001
        self.class_weights = torch.tensor([1.0, 2.0])  # Give more weight to positive class
        self.batch_size = 32
        self.window_size = 16
        self.epochs = 100
        self.patience = 15
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CryptoDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        assert len(self.sequences) == len(self.labels), "Sequences and labels must have same length"
        self.labels = np.array(self.labels)  # Convert to numpy array for consistent shape

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.FloatTensor([self.labels[idx]])  # Ensure label is a tensor of shape (1,)
        return sequence, label

def prepare_combined_data(processor):
    """Prepare combined training and testing datasets for both Bitcoin and Gala."""
    # Prepare Bitcoin data
    btc_train_data, btc_train_labels, btc_test_data, btc_test_labels = processor.prepare_dataset('bitcoin')
    
    # Prepare Gala data
    gala_train_data, gala_train_labels, gala_test_data, gala_test_labels = processor.prepare_dataset('gala')
    
    # Combine training data
    train_data = np.concatenate([btc_train_data, gala_train_data], axis=0)
    train_labels = np.concatenate([btc_train_labels, gala_train_labels], axis=0)
    
    # Combine test data
    test_data = np.concatenate([btc_test_data, gala_test_data], axis=0)
    test_labels = np.concatenate([btc_test_labels, gala_test_labels], axis=0)
    
    # Create datasets
    train_dataset = CryptoDataset(train_data, train_labels)
    test_dataset = CryptoDataset(test_data, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(train_loader, test_loader, config):
    """Train the trading model."""
    # Initialize model
    model = TradingTransformer(config)
    model = model.to(config.device)
    
    # Calculate class weights
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    train_labels = np.array(train_labels)
    
    # Calculate class weights
    pos_weight = len(train_labels) / (2 * np.sum(train_labels))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(config.device))
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # Early stopping
    best_val_loss = float('inf')
    patience = config.patience
    epochs_without_improvement = 0
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training
        for inputs, labels in train_loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            
            # Forward pass
            outputs = model(inputs)
            # Ensure labels have shape (batch_size,)
            labels = labels.squeeze(-1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate training accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)
                
                outputs = model(inputs)
                # Ensure labels have shape (batch_size,)
                labels = labels.squeeze(-1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate validation accuracy
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Store predictions and labels for metrics
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Convert to numpy arrays for metrics
        all_preds = np.array(all_probs) > 0.5
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate additional metrics
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # Print metrics
        print(f'Epoch [{epoch+1}/{config.epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(test_loader):.4f}',
              f' Acc: {val_acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}')
        
        # Save best model
        val_loss_epoch = val_loss / len(test_loader)
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with validation loss: {best_val_loss:.4f}')
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                print(f'Early stopping after {config.patience} epochs without improvement')
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model

def evaluate_model_performance(model, test_loader, crypto_name, device):
    """Evaluate model performance on a specific cryptocurrency."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            
            # Convert probabilities to binary predictions
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Convert predictions to binary
    pred_labels = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, pred_labels)
    precision = precision_score(all_labels, pred_labels, average='binary', zero_division=0)
    recall = recall_score(all_labels, pred_labels, average='binary', zero_division=0)
    f1 = f1_score(all_labels, pred_labels, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Print metrics
    print(f"\n{crypto_name} Performance Metrics:")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")
    print(f"- F1 Score: {f1:.4f}")
    print(f"- AUC: {auc:.4f}")
    
    # Calculate and print confusion matrix
    cm = confusion_matrix(all_labels, pred_labels)
    print("\nConfusion Matrix:")
    print("\tPredicted\tActual")
    print(f"TP\t{cm[0,0]}\tFP\t{cm[0,1]}")
    print(f"FN\t{cm[1,0]}\tTN\t{cm[1,1]}")
    
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
    """Main function to run the training."""
    # Initialize configuration
    config = TradingConfig()
    
    # Initialize data processor with new window size
    processor = DataProcessor(window_size=config.window_size)
    
    # Prepare data loaders
    train_loader, test_loader = prepare_combined_data(processor)
    
    # Train model
    trained_model = train_model(train_loader, test_loader, config)
    
    # Load the best model
    trained_model.load_state_dict(torch.load('best_model.pth'))
    trained_model.eval()
    
    # Evaluate model on both Bitcoin and Gala
    print("\nBitcoin Performance Metrics:")
    evaluate_model_performance(trained_model, test_loader, 'bitcoin', config.device)
    
    print("\nGala Performance Metrics:")
    evaluate_model_performance(trained_model, test_loader, 'gala', config.device)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
