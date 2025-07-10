import pandas as pd
import numpy as np
from datetime import datetime

# Simple backtest: Buy if model predicts 1, flat otherwise. No leverage, no transaction costs.
def backtest_with_predictions(csv_path, preds, window_size=16):
    df = pd.read_csv(csv_path)
    df = df.iloc[window_size:window_size+len(preds)]  # Align with preds
    df = df.copy()
    df['pred'] = preds
    # Calculate returns (next close / current close - 1)
    df['future_close'] = df['close'].shift(-1)
    df['return'] = df['future_close'] / df['close'] - 1
    # Strategy: Only take position if pred==1
    df['strategy_return'] = df['return'] * (df['pred'] == 1)
    df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
    df['cum_hold'] = (1 + df['return']).cumprod()
    # Metrics
    total_return = df['cum_strategy'].iloc[-2] - 1  # -2 to avoid last NaN
    hold_return = df['cum_hold'].iloc[-2] - 1
    trades = df['pred'].sum()
    win_trades = ((df['strategy_return'] > 0) & (df['pred'] == 1)).sum()
    win_rate = win_trades / trades if trades > 0 else 0
    print(f"Backtest Results:")
    print(f"Strategy Total Return: {total_return*100:.2f}%")
    print(f"Buy-and-Hold Return: {hold_return*100:.2f}%")
    print(f"Number of Trades: {trades}")
    print(f"Win Rate: {win_rate*100:.2f}%")
    # Save for plotting
    df[['timestamp', 'cum_strategy', 'cum_hold']].to_csv('backtest_equity_curve.csv', index=False)
    return df

if __name__ == '__main__':
    # Example: load predictions from walk-forward validation last fold
    import pickle
    with open('walk_forward_preds.pkl', 'rb') as f:
        preds = pickle.load(f)
    backtest_with_predictions('x:/stone/data/processed/bitcoin_usdt_15m_enriched_thresh0005_scaled.csv', preds, window_size=16)
