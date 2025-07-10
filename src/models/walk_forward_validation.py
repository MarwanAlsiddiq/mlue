import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformer_model import TradingConfig, TradingTransformer

# --- Walk-forward split utility ---
def walk_forward_split(X, y, n_splits=5, test_size=0.2):
    splits = []
    n = len(X)
    test_len = int(n * test_size)
    fold_size = (n - test_len) // n_splits
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_start = train_end
        test_end = test_start + test_len
        if test_end > n or train_end == 0:
            break
        splits.append((slice(0, train_end), slice(test_start, test_end)))
    return splits

# --- Data Loading ---
def load_data(csv_path, window_size=16, max_samples=10000):
    df = pd.read_csv(csv_path)
    features = [c for c in df.columns if c not in ['label', 'timestamp']]
    X = df[features].values.astype(np.float32)
    y = df['label'].values.astype(np.float32)
    X_seq = []
    y_seq = []
    for i in range(window_size, len(X)):
        X_seq.append(X[i-window_size:i])
        y_seq.append(y[i])
    X_seq = np.stack(X_seq)
    y_seq = np.array(y_seq)
    if len(X_seq) > max_samples:
        X_seq = X_seq[:max_samples]
        y_seq = y_seq[:max_samples]
    return X_seq, y_seq

# --- Main Walk-Forward Validation ---
def main():
    csv_path = 'x:/stone/data/processed/bitcoin_usdt_15m_enriched_thresh0005_scaled.csv'
    X, y = load_data(csv_path, window_size=16, max_samples=10000)
    splits = walk_forward_split(X, y, n_splits=5, test_size=0.2)
    results = []
    all_test_preds = []
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"\n=== Fold {i+1} ===")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)
        config = TradingConfig()
        config.input_dim = X.shape[2]
        config.window_size = 16
        model = TradingTransformer(config)
        model = model.to(config.device)
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=config.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        best_loss = float('inf')
        patience = 4
        epochs_no_improve = 0
        for epoch in range(config.epochs):
            model.train()
            losses = []
            for xb, yb in train_loader:
                xb = xb.to(config.device)
                yb = yb.to(config.device)
                optimizer.zero_grad()
                logits = model(xb).float()
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            avg_loss = np.mean(losses)
            print(f"Fold {i+1} Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        # --- Evaluation ---
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(config.device)
                logits = model(xb).float()
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        print(f'Fold {i+1} Test Accuracy: {acc:.4f}')
        print(f'Fold {i+1} Test Macro F1: {f1:.4f}')
        print(f'Fold {i+1} Test Macro Precision: {prec:.4f}')
        print(f'Fold {i+1} Test Macro Recall: {rec:.4f}')
        print('Confusion Matrix:')
        print(cm)
        results.append({'fold': i+1, 'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec, 'cm': cm})
        all_test_preds.append(np.array(all_preds))
    print("\n=== Walk-Forward Validation Results ===")
    for r in results:
        print(f"Fold {r['fold']}: Acc={r['acc']:.4f}, F1={r['f1']:.4f}, Prec={r['prec']:.4f}, Rec={r['rec']:.4f}")
        print(r['cm'])
    # Save concatenated predictions for backtesting
    from save_walk_forward_preds import save_preds
    save_preds(all_test_preds, filename='walk_forward_preds.pkl')

if __name__ == '__main__':
    main()
