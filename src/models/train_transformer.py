import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformer_model import TradingConfig, TradingTransformer

# --- Hyperparameters ---
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005
WINDOW_SIZE = 16

# --- Data Loading and Preprocessing ---
def load_data(csv_path, window_size=16):
    df = pd.read_csv(csv_path)
    features = [c for c in df.columns if c not in ['label', 'timestamp']]
    X = df[features].values.astype(np.float32)
    y = df['label'].values.astype(np.float32)
    # Create rolling windows
    X_seq = []
    y_seq = []
    for i in range(window_size, len(X)):
        X_seq.append(X[i-window_size:i])
        y_seq.append(y[i])
    X_seq = np.stack(X_seq)
    y_seq = np.array(y_seq)
    return X_seq, y_seq

def main():
    # You can change the path here to try different datasets
    csv_path = 'x:/stone/data/processed/bitcoin_usdt_15m_enriched_thresh0005_scaled.csv'
    X, y = load_data(csv_path, window_size=WINDOW_SIZE)
    # Use a 10,000-sample subset for fast diagnostics
    max_samples = 10000
    if len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    config = TradingConfig()
    config.input_dim = X.shape[2]
    config.window_size = WINDOW_SIZE
    model = TradingTransformer(config)
    model = model.to(config.device)

    # Compute pos_weight for class imbalance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=config.device)
    print(f"Train class distribution: 0={n_neg}, 1={n_pos}, pos_weight={pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    best_loss = float('inf')
    best_epoch = 0
    patience = 3
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
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
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best epoch: {best_epoch+1} | Best loss: {best_loss:.4f}")

    # --- Evaluation ---
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    all_probs = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(config.device)
            logits = model(xb).float()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Final Test Accuracy: {acc:.4f}')
    print(f'Final Test Macro F1: {f1:.4f}')
    print(f'Final Test Macro Precision: {prec:.4f}')
    print(f'Final Test Macro Recall: {rec:.4f}')
    print('Confusion Matrix:')
    print(cm)
    # Debug: Print unique values and first 20 logits/probabilities
    print('Unique logits:', np.unique(all_logits))
    print('Unique probabilities:', np.unique(all_probs))
    print('First 20 logits:', all_logits[:20])
    print('First 20 probabilities:', all_probs[:20])

if __name__ == '__main__':
    main()
