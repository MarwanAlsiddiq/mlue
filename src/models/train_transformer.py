import torch
import torch.nn as nn
import pandas as pd
import numpy as np
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
    csv_path = 'x:/stone/data/processed/bitcoin_usdt_15m_enriched_thresh0005.csv'
    X, y = load_data(csv_path, window_size=WINDOW_SIZE)
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
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {np.mean(losses):.4f}")

    # --- Evaluation ---
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(config.device)
            logits = model(xb).float()
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Accuracy: {acc:.4f}')
    print(f'Macro F1: {f1:.4f}')
    print(f'Macro Precision: {prec:.4f}')
    print(f'Macro Recall: {rec:.4f}')
    print('Confusion Matrix:')
    print(cm)

if __name__ == '__main__':
    main()
