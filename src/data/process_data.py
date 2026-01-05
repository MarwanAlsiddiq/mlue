import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os

class DataProcessor:
    def __init__(self, window_size=16, data_dir='data/processed'):
        """Initialize the data processor."""
        self.window_size = window_size
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
        """Load and process cryptocurrency data."""
        if crypto_name == 'bitcoin':
            df = pd.read_csv('data/raw/btcusdt_data.csv')
            df['marketcap'] = df['quote_volume']  # For Bitcoin, use quote volume as marketcap
        elif crypto_name == 'gala':
            df = pd.read_csv('data/raw/galausdt_data.csv')
            df['marketcap'] = df['quote_volume']  # For Gala, use quote volume as marketcap
        else:
            raise ValueError(f"Unsupported cryptocurrency: {crypto_name}")

        # Calculate technical indicators
        # RSI (14 periods)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Drop any remaining NaN values from technical indicators
        df = df.dropna()

        # Extract relevant features
        features = df[['open', 'high', 'low', 'close', 'volume', 'marketcap',
                     'rsi', 'macd', 'macd_signal', 'macd_hist']].values

        return features
        
    def add_technical_indicators(self, df):
        """Add RSI and MACD indicators to the dataframe."""
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df

    def create_sequences(self, data):
        """Create sequences of data with window size."""
        # Create DataFrame with all features
        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume', 'marketcap',
                                       'rsi', 'macd', 'macd_signal', 'macd_hist'])
        
        # Drop NaN values from technical indicators
        df = df.dropna()
        
        # Normalize the data
        normalized = self.scaler.fit_transform(df.values)
        
        # Convert back to DataFrame with same columns
        df = pd.DataFrame(normalized, columns=['open', 'high', 'low', 'close', 'volume', 'marketcap',
                                             'rsi', 'macd', 'macd_signal', 'macd_hist'])
        
        # Ensure we have enough data for sequences
        if len(df) < self.window_size:
            raise ValueError(f"Not enough data points after processing. Need at least {self.window_size} points")
        
        sequences = []
        for i in range(len(df) - self.window_size):
            # Create sequence of window_size length
            seq = df.iloc[i:i + self.window_size].values
            sequences.append(seq)
        
        # Convert to numpy array
        sequences = np.array(sequences)
        
        # Create labels (1 if price increases after window, 0 if decreases)
        # Use the close price at the end of the sequence vs the next close price
        labels = (df['close'].iloc[self.window_size:].values > df['close'].iloc[self.window_size-1:-1].values).astype(int)
        
        return sequences, labels

    def prepare_dataset(self, crypto_name='bitcoin', test_size=0.2):
        """Prepare training and testing datasets for a specific cryptocurrency"""
        # Load cryptocurrency data
        data = self.load_crypto_data(crypto_name)
        
        # Create sequences and labels
        sequences, labels = self.create_sequences(data)
        
        # Split into train and test sets
        split_idx = int(len(sequences) * (1 - test_size))
        
        return (
            sequences[:split_idx],
            labels[:split_idx],
            sequences[split_idx:],
            labels[split_idx:]
        )
