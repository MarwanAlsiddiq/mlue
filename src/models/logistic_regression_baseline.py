import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# --- Data Loading ---
def load_data(csv_path, window_size=16, sample_size=10000):
    df = pd.read_csv(csv_path)
    features = [c for c in df.columns if c not in ['label', 'timestamp']]
    X = df[features].values.astype(np.float32)
    y = df['label'].values.astype(np.float32)
    # Rolling window alignment (match Transformer)
    X_seq = []
    y_seq = []
    for i in range(window_size, len(X)):
        X_seq.append(X[i-window_size:i].flatten())
        y_seq.append(y[i])
    X_seq = np.stack(X_seq)
    y_seq = np.array(y_seq)
    # Use a subset for speed
    if sample_size is not None and len(X_seq) > sample_size:
        X_seq = X_seq[:sample_size]
        y_seq = y_seq[:sample_size]
    return X_seq, y_seq

def main():
    csv_path = 'x:/stone/data/processed/bitcoin_usdt_15m_enriched_thresh0005.csv'
    X, y = load_data(csv_path, window_size=16, sample_size=10000)
    print(f"Feature shape: {X.shape}, Label shape: {y.shape}")
    print(f"Label distribution: {np.bincount(y.astype(int))}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print(f'Macro F1: {f1:.4f}')
    print(f'Macro Precision: {prec:.4f}')
    print(f'Macro Recall: {rec:.4f}')
    print('Confusion Matrix:')
    print(cm)
    print('First 20 predictions:', y_pred[:20])
    print('First 20 true labels:', y_test[:20])

if __name__ == '__main__':
    main()
