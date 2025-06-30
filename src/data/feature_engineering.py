import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator

def enrich_features(df):
    # Only keep necessary columns
    df = df[['timestamp','open','high','low','close','volume','quote_volume']].copy()
    # EMA/SMA
    if len(df) > 20:
        df['ema_10'] = EMAIndicator(df['close'], window=10, fillna=True).ema_indicator()
        df['ema_20'] = EMAIndicator(df['close'], window=20, fillna=True).ema_indicator()
        df['sma_10'] = SMAIndicator(df['close'], window=10, fillna=True).sma_indicator()
        df['sma_20'] = SMAIndicator(df['close'], window=20, fillna=True).sma_indicator()
    # Bollinger Bands
    if len(df) > 21:
        bb = BollingerBands(df['close'], window=20, fillna=True)
        df['bb_bbm'] = bb.bollinger_mavg()
        df['bb_bbh'] = bb.bollinger_hband()
        df['bb_bbl'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
    # ATR
    if len(df) > 15:
        df['atr_14'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14, fillna=True).average_true_range()
    # RSI
    if len(df) > 15:
        df['rsi_14'] = RSIIndicator(df['close'], window=14, fillna=True).rsi()
    # MACD
    if len(df) > 26:
        macd = MACD(df['close'], fillna=True)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
    # OBV
    if len(df) > 2:
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume'], fillna=True).on_balance_volume()
    # Lagged returns
    df['return_1'] = df['close'].pct_change(1)
    df['return_3'] = df['close'].pct_change(3)
    df['return_5'] = df['close'].pct_change(5)
    # Rolling volatility
    df['volatility_5'] = df['close'].rolling(5).std()
    df['volatility_10'] = df['close'].rolling(10).std()
    # Time features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df = df.fillna(0)
    return df

def create_labels(df, window_size=16, threshold=0.005):
    # Label: 1 if next close > last close in window by at least threshold (default 0.5%)
    future_close = df['close'].shift(-window_size)
    last_close = df['close']
    labels = ((future_close - last_close) / last_close > threshold).astype(int)
    return labels

def process_crypto(symbol, input_csv, output_csv, window_size=16):
    df = pd.read_csv(input_csv)
    df = enrich_features(df)
    df['label'] = create_labels(df, window_size)
    # Drop last window_size rows (no label)
    df = df.iloc[:-window_size]
    df.to_csv(output_csv, index=False)
    print(f"Saved enriched {symbol} data to {output_csv}")

if __name__ == "__main__":
    process_crypto('bitcoin', '../../data/raw/btcusdt_data.csv', '../../data/processed/btcusdt_enriched.csv')
    process_crypto('gala', '../../data/raw/galausdt_data.csv', '../../data/processed/galausdt_enriched.csv')
