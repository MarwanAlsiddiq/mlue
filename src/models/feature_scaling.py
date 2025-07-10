import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_features(input_csv, output_csv, window_size=16):
    df = pd.read_csv(input_csv)
    features = [c for c in df.columns if c not in ['label', 'timestamp']]
    scaler = StandardScaler()
    # Only scale features, not labels/timestamps
    vals = scaler.fit_transform(df[features].values)
    scaled_df = pd.DataFrame(vals, columns=features, index=df.index)
    scaled_df['label'] = df['label']
    scaled_df['timestamp'] = df['timestamp']
    scaled_df = scaled_df[['timestamp'] + features + ['label']]
    scaled_df.to_csv(output_csv, index=False)
    print(f"Saved scaled features to {output_csv}")

if __name__ == '__main__':
    scale_features('x:/stone/data/processed/bitcoin_usdt_15m_enriched_thresh0005.csv',
                  'x:/stone/data/processed/bitcoin_usdt_15m_enriched_thresh0005_scaled.csv')
