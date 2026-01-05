import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Utility: Train and evaluate on a given dataset

def run_lgbm(csv_path, crypto_name):
    print(f'\n==== {crypto_name.upper()} ===')
    df = pd.read_csv(csv_path)
    y = df['label']
    X = df.drop(['label', 'timestamp'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Oversample minority class in training set
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    print(f"After oversampling: {dict(pd.Series(y_train_res).value_counts())}")

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        random_state=42
    )
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print('Confusion Matrix:')
    print(cm)

if __name__ == "__main__":
    run_lgbm('../../data/processed/btcusdt_enriched.csv', 'bitcoin')
    run_lgbm('../../data/processed/galausdt_enriched.csv', 'gala')
