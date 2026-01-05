import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# --- Parameters ---
TAKE_PROFIT = 0.005  # 0.5%
STOP_LOSS = -0.005   # -0.5%
WINDOW_SIZE = 16

# --- Utility: Backtest trading strategy based on model predictions ---
def backtest(csv_path, model_path=None, initial_balance=1000):
    df = pd.read_csv(csv_path)
    y = df['label']
    X = df.drop(['label', 'timestamp'], axis=1)
    # Use last 20% as test set
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    prices = df['close'].iloc[split_idx:].values
    if model_path:
        model = lgb.Booster(model_file=model_path)
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1) - 1  # Convert to [-1,0,1]
    else:
        # For quick demo, retrain model on train set
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        model = lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=200, learning_rate=0.03, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    # Backtest logic
    balance = initial_balance
    equity_curve = [balance]
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0
    trade_log = []
    for i in range(len(y_pred) - WINDOW_SIZE):
        signal = y_pred[i]
        price = prices[i]
        # Only enter if flat
        if position == 0:
            if signal == 1:
                position = 1
                entry_price = price
                trade_log.append({'type': 'long', 'entry': price, 'idx': i})
            elif signal == -1:
                position = -1
                entry_price = price
                trade_log.append({'type': 'short', 'entry': price, 'idx': i})
        # Manage open position
        elif position == 1:
            # Check take-profit/stop-loss in next WINDOW_SIZE bars
            future_prices = prices[i+1:i+1+WINDOW_SIZE]
            tp_hit = np.any((future_prices - entry_price) / entry_price >= TAKE_PROFIT)
            sl_hit = np.any((future_prices - entry_price) / entry_price <= STOP_LOSS)
            if tp_hit:
                pnl = entry_price * TAKE_PROFIT
                balance += pnl
                position = 0
                trade_log[-1].update({'exit': entry_price * (1+TAKE_PROFIT), 'pnl': pnl, 'result': 'tp'})
            elif sl_hit:
                pnl = entry_price * STOP_LOSS
                balance += pnl
                position = 0
                trade_log[-1].update({'exit': entry_price * (1+STOP_LOSS), 'pnl': pnl, 'result': 'sl'})
        elif position == -1:
            # Short: TP/SL logic reversed
            future_prices = prices[i+1:i+1+WINDOW_SIZE]
            tp_hit = np.any((entry_price - future_prices) / entry_price >= TAKE_PROFIT)
            sl_hit = np.any((entry_price - future_prices) / entry_price <= STOP_LOSS)
            if tp_hit:
                pnl = entry_price * TAKE_PROFIT
                balance += pnl
                position = 0
                trade_log[-1].update({'exit': entry_price * (1-TAKE_PROFIT), 'pnl': pnl, 'result': 'tp'})
            elif sl_hit:
                pnl = entry_price * STOP_LOSS
                balance += pnl
                position = 0
                trade_log[-1].update({'exit': entry_price * (1-STOP_LOSS), 'pnl': pnl, 'result': 'sl'})
        equity_curve.append(balance)
    # Final stats
    trade_results = pd.DataFrame(trade_log)
    if trade_results.empty:
        print("No trades executed. Model did not generate any actionable signals.")
        return equity_curve, trade_results
    n_trades = len(trade_results)
    n_wins = (trade_results['result'] == 'tp').sum()
    n_losses = (trade_results['result'] == 'sl').sum()
    win_rate = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0
    max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve)
    print(f"Total trades: {len(trade_results)} | Wins: {n_wins} | Losses: {n_losses} | Win rate: {win_rate:.2%}")
    print(f"Total PnL: {equity_curve[-1] - equity_curve[0]:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}")
    print(trade_results)
    return equity_curve, trade_results

if __name__ == "__main__":
    # Example usage
    equity, trades = backtest('data/processed/bitcoin_usdt_15m_enriched.csv')
