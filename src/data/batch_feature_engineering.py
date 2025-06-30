import os
import pandas as pd
from feature_engineering import enrich_features, create_labels

agg_dir = '../../data/agg'
out_dir = '../../data/processed'
os.makedirs(out_dir, exist_ok=True)

window_size = 16
threshold = 0.005  # 0.5%

for fname in os.listdir(agg_dir):
    if not fname.endswith('.csv'):
        continue
    coin_tf = fname.replace('.csv','')
    in_path = os.path.join(agg_dir, fname)
    out_path = os.path.join(out_dir, f'{coin_tf}_enriched.csv')
    df = pd.read_csv(in_path)
    if len(df) < window_size + 2:
        continue
    df_feat = enrich_features(df)
    df_feat['label'] = create_labels(df_feat, window_size, threshold)
    df_feat = df_feat.iloc[:-window_size]
    df_feat.to_csv(out_path, index=False)
    print(f'Saved {out_path}')
