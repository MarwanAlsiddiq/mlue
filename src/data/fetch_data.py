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

def fetch_and_save_crypto_data(symbol, interval='15m', start_time=None, end_time=None, data_dir='data/raw'):
    """Fetch and save all available crypto data for a symbol/interval between start_time and end_time."""
    os.makedirs(data_dir, exist_ok=True)
    fetcher = BinanceDataFetcher(symbol, interval, 1000)
    all_dfs = []
    # Convert to timestamps
    start_ts = int(pd.Timestamp(start_time).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_time).timestamp() * 1000)
    curr_start = start_ts
    while curr_start < end_ts:
        df = fetcher.fetch_data(pd.to_datetime(curr_start, unit='ms'), pd.to_datetime(end_ts, unit='ms'))
        if df.empty:
            break
        all_dfs.append(df)
        last_ts = int(df.index[-1].timestamp() * 1000)
        if last_ts == curr_start:
            break
        curr_start = last_ts + 1
        time.sleep(1)  # Respect API rate limits
    if all_dfs:
        full_df = pd.concat(all_dfs)
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        filename = os.path.join(data_dir, f"{symbol.lower()}_{interval}_full.csv")
        fetcher.save_to_csv(full_df, filename)
        return full_df
    else:
        print("No data fetched.")
        return pd.DataFrame()

if __name__ == "__main__":
    import datetime
    import time
    # Define intervals and max historical days (approximate, Binance limits)
    intervals = {
        '1m': 90,    # 3 months
        '5m': 365,   # 1 year
        '15m': 365,  # 1 year
        '30m': 730,  # 2 years
        '1h': 1095,  # 3 years
        '4h': 1825   # 5 years
    }
    symbols = ['BTCUSDT', 'GALAUSDT']
    end = datetime.datetime.now()
    for interval, days in intervals.items():
        start = end - datetime.timedelta(days=days)
        for symbol in symbols:
            print(f"Fetching {symbol} {interval} data from {start} to {end}")
            fetch_and_save_crypto_data(
                symbol=symbol,
                interval=interval,
                start_time=start.strftime('%Y-%m-%d %H:%M:%S'),
                end_time=end.strftime('%Y-%m-%d %H:%M:%S'),
                data_dir='data/raw'
            )
            time.sleep(2)  # Be gentle to the API
