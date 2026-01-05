import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

proc_dir = '../../data/processed'
window_size = 16

for fname in os.listdir(proc_dir):
    if not fname.endswith('_enriched.csv'):
        continue
    coin_tf = fname.replace('_enriched.csv','')
    print(f'\n==== {coin_tf.upper()} ====')
    df = pd.read_csv(os.path.join(proc_dir, fname))
    if 'label' not in df.columns or df['label'].nunique() < 2:
        print('Not enough label variety, skipping.')
        continue
    y = df['label']
    X = df.drop(['label', 'timestamp'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    print(f'After oversampling: {dict(pd.Series(y_train_res).value_counts())}')
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, random_state=42)
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print('Confusion Matrix:')
    print(cm)
