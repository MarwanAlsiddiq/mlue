import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os

class DataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Load and preprocess raw financial data"""
        # Load sample data
        df = pd.read_csv(os.path.join(self.data_dir, 'sample_data.csv'))
        
        # Extract relevant features
        features = df[['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']].values
        
        return features
        
    def load_crypto_data(self, crypto_name):
        """Load cryptocurrency data from processed folder and calculate technical indicators"""
        # Try both naming conventions
        usdt_file = os.path.join(self.data_dir, f'{crypto_name}_usdt_data_processed.csv')
        regular_file = os.path.join(self.data_dir, f'{crypto_name}_processed.csv')
        
        # Load the processed CSV
        if os.path.exists(usdt_file):
            df = pd.read_csv(usdt_file)
        elif os.path.exists(regular_file):
            df = pd.read_csv(regular_file)
        else:
            raise FileNotFoundError(f"Could not find data file for {crypto_name} in {self.data_dir}")
        
        # Calculate technical indicators using pure pandas
        # Calculate RSI (14 periods)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Calculate EMA (20 periods)
        df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Replace NaN values with 0
        df['RSI'] = df['RSI'].fillna(0)
        df['MACD'] = df['MACD'].fillna(0)
        df['MACD_Signal'] = df['MACD_Signal'].fillna(0)
        df['MACD_Hist'] = df['MACD_Hist'].fillna(0)
        df['EMA'] = df['EMA'].fillna(0)
        
        # Drop NaN values resulting from indicator calculations
        df = df.dropna()
        
        # Extract relevant features based on crypto type
        if crypto_name.lower() == 'gala':
            features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                         'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'EMA']].values
        else:
            features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap',
                         'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'EMA']].values
        
        return features
        
    def create_sequences(self, data, window_size=8):
        """Create sequences of time series data for Transformer"""
        sequences = []
        
        for i in range(len(data) - window_size):
            window = data[i:i + window_size]
            # Normalize the window data
            window = self.scaler.fit_transform(window)
            
            # Convert to float32
            window = window.astype(np.float32)
            
            # Add to list
            sequences.append(window)
            
        # Convert to numpy array
        sequences_array = np.array(sequences)
        
        return sequences_array
        
    def prepare_dataset(self, crypto_name, window_size=8, test_size=0.2):
        """Prepare training and testing datasets for a specific cryptocurrency"""
        # Load cryptocurrency data
        data = self.load_crypto_data(crypto_name)
        
        # Create sequences
        sequences = self.create_sequences(data, window_size)
        
        # Create labels from sequences (1 if price increases after window, 0 if decreases)
        labels = (data[window_size:, 3] > data[window_size-1:-1, 3]).astype(int)
        
        # Split into train and test sets
        split_idx = int(len(sequences) * (1 - test_size))
        
        return (
            sequences[:split_idx],
            labels[:split_idx],
            sequences[split_idx:],
            labels[split_idx:]
        )
        return train_tensors, train_labels, test_tensors, test_labels
