import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
import matplotlib.pyplot as plt
import json
from typing import Dict, Any
import random

from .transformer_model import TradingTransformer, TradingConfig
from ..data.feature_engineering import enrich_features, create_labels

class TradingDataset(Dataset):
    def __init__(self, features, targets, sequence_length=16):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            self.features[idx:idx + self.sequence_length],
            self.targets[idx + self.sequence_length]
        )

class EnhancedTransformerTrainer:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.device = config.device
        self.model = TradingTransformer(config).to(self.device)
        self.scaler = StandardScaler()
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
    def prepare_data(self, data_path: str, mode: str = 'regression'):
        """Load and prepare data for training"""
        df = pd.read_csv(data_path)
        
        # Enrich features
        df = enrich_features(df)
        
        # Create labels
        labels = create_labels(df, window_size=self.config.window_size, mode=mode)
        
        # Remove rows with NaN labels
        valid_idx = ~labels.isna()
        df = df[valid_idx]
        labels = labels[valid_idx]
        
        # Select feature columns (exclude timestamp and label)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
        features = df[feature_cols].values
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.config.window_size):
            X.append(features[i:i + self.config.window_size])
            y.append(labels.iloc[i + self.config.window_size])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None if mode == 'regression' else y
        )
        
        return X_train, X_val, y_train, y_val
    
    def get_loss_function(self, mode: str):
        """Get appropriate loss function based on task type"""
        if mode == 'regression':
            return nn.MSELoss()
        elif mode == '3class':
            # Convert labels from [-1, 0, 1] to [0, 1, 2]
            class_weights = self.config.class_weights.to(self.device)
            return nn.CrossEntropyLoss(weight=class_weights)
        elif mode == 'binary':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def train_epoch(self, train_loader, optimizer, criterion, mode: str):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(batch_features)
            
            # Handle different task types
            if mode == 'regression':
                loss = criterion(outputs, batch_targets)
                predictions.extend(outputs.detach().cpu().numpy())
                targets.extend(batch_targets.detach().cpu().numpy())
            elif mode == '3class':
                # Convert targets from [-1, 0, 1] to [0, 1, 2]
                batch_targets = (batch_targets + 1).long()
                loss = criterion(outputs, batch_targets)
                predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
                targets.extend(batch_targets.detach().cpu().numpy())
            elif mode == 'binary':
                loss = criterion(outputs, batch_targets.unsqueeze(1))
                predictions.extend((torch.sigmoid(outputs) > 0.5).float().detach().cpu().numpy())
                targets.extend(batch_targets.detach().cpu().numpy())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader), predictions, targets
    
    def validate(self, val_loader, criterion, mode: str):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_features)
                
                # Handle different task types
                if mode == 'regression':
                    loss = criterion(outputs, batch_targets)
                    predictions.extend(outputs.detach().cpu().numpy())
                    targets.extend(batch_targets.detach().cpu().numpy())
                elif mode == '3class':
                    batch_targets = (batch_targets + 1).long()
                    loss = criterion(outputs, batch_targets)
                    predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
                    targets.extend(batch_targets.detach().cpu().numpy())
                elif mode == 'binary':
                    loss = criterion(outputs, batch_targets.unsqueeze(1))
                    predictions.extend((torch.sigmoid(outputs) > 0.5).float().detach().cpu().numpy())
                    targets.extend(batch_targets.detach().cpu().numpy())
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader), predictions, targets
    
    def train(self, data_path: str, mode: str = 'regression', save_dir: str = 'models'):
        """Main training function"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(data_path, mode)
        
        # Create datasets and loaders
        train_dataset = TradingDataset(X_train, y_train, self.config.window_size)
        val_dataset = TradingDataset(X_val, y_val, self.config.window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Setup training
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = self.get_loss_function(mode)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print(f"Training {mode} model for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            train_loss, train_preds, train_targets = self.train_epoch(train_loader, optimizer, criterion, mode)
            val_loss, val_preds, val_targets = self.validate(val_loader, criterion, mode)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Print metrics
            if mode == 'regression':
                train_mae = mean_absolute_error(train_targets, train_preds)
                val_mae = mean_absolute_error(val_targets, val_preds)
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, MAE={train_mae:.4f} | Val Loss={val_loss:.4f}, MAE={val_mae:.4f}")
            else:
                train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
                val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'scaler': self.scaler,
                    'mode': mode
                }, os.path.join(save_dir, f'best_{mode}_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'mode': mode,
            'config': self.config.__dict__
        }
        
        with open(os.path.join(save_dir, f'{mode}_training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'{mode.capitalize()} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{mode}_training_curve.png'))
        plt.close()
        
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        return history

def main():
    # Configuration
    config = TradingConfig()
    config.task_type = 'regression'  # Change to '3class' or 'binary' as needed
    config.epochs = 50
    config.batch_size = 64
    
    # Initialize trainer
    trainer = EnhancedTransformerTrainer(config)
    
    # Train model
    data_path = '../../data/processed/btcusdt_enriched.csv'
    history = trainer.train(data_path, mode=config.task_type, save_dir='models/enhanced_transformer')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
