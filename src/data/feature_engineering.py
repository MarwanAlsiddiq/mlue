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
    # Rolling max/min for trade simulation
    df['future_max_8'] = df['close'].shift(-1).rolling(window=8).max().shift(1-8)
    df['future_min_8'] = df['close'].shift(-1).rolling(window=8).min().shift(1-8)
    df['future_max_16'] = df['close'].shift(-1).rolling(window=16).max().shift(1-16)
    df['future_min_16'] = df['close'].shift(-1).rolling(window=16).min().shift(1-16)
    # Time features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df = df.fillna(0)
    return df

def create_labels(df, window_size=1, threshold=0.002):
    """
    Binary label for trading:
    1: if next close >= threshold above current close
    0: otherwise
    """
    arr = df['close'].values
    labels = np.zeros(len(df), dtype=int)
    for i in range(len(df) - 1):
        curr = arr[i]
        next_close = arr[i+1]
        if (next_close - curr) / curr >= threshold:
            labels[i] = 1
        else:
            labels[i] = 0
    return pd.Series(labels, index=df.index)

def process_crypto(symbol, input_csv, output_csv, window_size=16, threshold=0.002):
    df = pd.read_csv(input_csv)
    df = enrich_features(df)
    df['label'] = create_labels(df, window_size, threshold)
    # Drop last window_size rows (no label)
    df = df.iloc[:-window_size]
    df.to_csv(output_csv, index=False)
    print(f"Saved enriched {symbol} data to {output_csv}")

if __name__ == "__main__":
    process_crypto('bitcoin', '../../data/raw/btcusdt_data.csv', '../../data/processed/btcusdt_enriched.csv')
    process_crypto('gala', '../../data/raw/galausdt_data.csv', '../../data/processed/galausdt_enriched.csv')
