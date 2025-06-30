import pandas as pd
import os

timeframes = {
    '15m': '15T',
    '30m': '30T',
    '1h': '1H',
    '4h': '4H',
}

raw_dir = '../../data/raw'
out_dir = '../../data/agg'
os.makedirs(out_dir, exist_ok=True)

# Columns to aggregate
agg_dict = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'quote_volume': 'sum',
}

for fname in os.listdir(raw_dir):
    if not fname.endswith('.csv') or 'test' in fname or 'sample' in fname:
        continue
    coin = fname.split('_')[0].replace('usdt','').replace('btc','bitcoin').replace('gala','gala')
    fpath = os.path.join(raw_dir, fname)
    df = pd.read_csv(fpath)
    if 'timestamp' not in df.columns:
        continue
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')
    for tf_name, tf_rule in timeframes.items():
        df_agg = df.resample(tf_rule).agg(agg_dict)
        df_agg = df_agg.dropna().reset_index()
        outname = f"{coin}_usdt_{tf_name}.csv"
        outpath = os.path.join(out_dir, outname)
        df_agg.to_csv(outpath, index=False)
        print(f"Saved {outpath}")
