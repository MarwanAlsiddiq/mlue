import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os

class BinanceDataFetcher:
    def __init__(self, symbol, interval='1m', limit=1000):
        """Initialize the Binance data fetcher.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Time interval (e.g., '1m', '5m', '1h')
            limit (int): Number of records to fetch (max 1000)
        """
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.base_url = 'https://api.binance.com/api/v3/klines'
        
    def fetch_data(self, start_time=None, end_time=None):
        """Fetch historical data from Binance API.
        
        Args:
            start_time (str, optional): Start time in format '2025-03-19 00:00:00'
            end_time (str, optional): End time in format '2025-03-19 00:00:00'
            
        Returns:
            pd.DataFrame: DataFrame containing the historical data
        """
        # Convert datetime strings to timestamps if provided
        if start_time:
            start_time = int(pd.Timestamp(start_time).timestamp() * 1000)
        if end_time:
            end_time = int(pd.Timestamp(end_time).timestamp() * 1000)
            
        # Fetch data in chunks if limit is exceeded
        all_data = []
        while True:
            params = {
                'symbol': self.symbol,
                'interval': self.interval,
                'limit': self.limit,
                'startTime': start_time,
                'endTime': end_time
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                    
                all_data.extend(data)
                
                # Get the timestamp of the last record for the next request
                start_time = int(data[-1][0]) + 1
                
                # Add delay to respect API rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
                
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert numeric columns to appropriate types
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                   'taker_buy_base', 'taker_buy_quote']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df

    def save_to_csv(self, df, filename):
        """Save DataFrame to CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Output filename
        """
        df.to_csv(filename, index=True)
        print(f"Data saved to {filename}")

def fetch_and_save_crypto_data(symbol, interval='1m', limit=1000, 
                              start_time=None, end_time=None, 
                              data_dir='data/raw'):
    """Fetch and save cryptocurrency data.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Time interval (e.g., '1m', '5m', '1h')
        limit (int): Number of records to fetch
        start_time (str): Start time in format '2025-03-19 00:00:00'
        end_time (str): End time in format '2025-03-19 00:00:00'
        data_dir (str): Directory to save the data
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize fetcher
    fetcher = BinanceDataFetcher(symbol, interval, limit)
    
    # Fetch data
    df = fetcher.fetch_data(start_time, end_time)
    
    # Save to CSV
    filename = os.path.join(data_dir, f"{symbol.lower()}_data.csv")
    fetcher.save_to_csv(df, filename)
    
    return df

if __name__ == "__main__":
    # Example usage
    # Fetch last 1000 minutes of BTCUSDT data
    df = fetch_and_save_crypto_data(
        symbol='BTCUSDT',
        interval='1m',
        limit=1000,
        start_time='2025-03-19 00:00:00',
        end_time='2025-03-19 23:59:59'
    )
    
    # Fetch last 1000 minutes of GALAUSDT data
    df = fetch_and_save_crypto_data(
        symbol='GALAUSDT',
        interval='1m',
        limit=1000,
        start_time='2025-03-19 00:00:00',
        end_time='2025-03-19 23:59:59'
    )
