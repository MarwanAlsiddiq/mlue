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
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        random_state=42,
        objective='multiclass',
        num_class=3
    )
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[-1,0,1])
    print(f'Accuracy: {acc:.4f}')
    print(f'Macro F1: {f1:.4f}')
    print(f'Macro Precision: {prec:.4f}')
    print(f'Macro Recall: {rec:.4f}')
    print('Confusion Matrix (rows: true, cols: pred):')
    print(cm)
    print('Class distribution in test:', dict(pd.Series(y_test).value_counts()))

    # Feature importance analysis
    feature_names = X.columns
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    print('Top 10 Feature Importances:')
    print(fi_df.head(10))
    # Save feature importances to CSV
    fi_path = os.path.join(proc_dir, f'{coin_tf}_feature_importance.csv')
    fi_df.to_csv(fi_path, index=False)
    print(f'Feature importances saved to {fi_path}')
