import pickle
import numpy as np

# This script will extract and save all predictions from walk_forward_validation.py
# Run this inside walk_forward_validation.py after each fold, then save all test predictions in order.

def save_preds(all_preds, filename='walk_forward_preds.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(np.concatenate(all_preds), f)
    print(f"Saved walk-forward predictions to {filename}")
