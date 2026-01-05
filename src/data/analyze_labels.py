import numpy as np
from process_data import DataProcessor

def main():
    processor = DataProcessor()
    for crypto in ['bitcoin', 'gala']:
        print(f'--- {crypto.upper()} ---')
        X_train, y_train, X_test, y_test = processor.prepare_dataset(crypto)
        print('Train label distribution:')
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f'  Label {u}: {c} ({c/len(y_train):.2%})')
        print('Test label distribution:')
        unique, counts = np.unique(y_test, return_counts=True)
        for u, c in zip(unique, counts):
            print(f'  Label {u}: {c} ({c/len(y_test):.2%})')

if __name__ == "__main__":
    main()
