# Enhanced Crypto Trading Model Training on Kaggle
# This script can be copied into a Kaggle notebook

# Cell 1: Install required packages
print("Installing required packages...")
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q pandas numpy scikit-learn matplotlib seaborn ta
!pip install -q tqdm

# Cell 2: Check GPU availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 3: Clone repository and setup
import os
import zipfile

# Option 1: Clone from GitHub (if public)
!git clone https://github.com/MarwanAlsiddiq/mlue.git /kaggle/working/mlue

# Option 2: Upload your project as a dataset and unzip
# with zipfile.ZipFile('/kaggle/input/mlue.zip', 'r') as zip_ref:
#     zip_ref.extractall('/kaggle/working/mlue')

# Change to project directory
os.chdir('/kaggle/working/mlue')
!ls -la

# Cell 4: Prepare data
import pandas as pd
import numpy as np
from src.data.feature_engineering import process_crypto, enrich_features, create_labels

# Process data for different modes
modes = ['regression', '3class']  # You can also add 'binary'
symbols = ['bitcoin', 'gala']

for mode in modes:
    for symbol in symbols:
        input_file = f'data/raw/{symbol}usdt_data.csv'
        output_file = f'data/processed/{symbol}usdt_enriched_{mode}.csv'
        
        print(f"Processing {symbol} for {mode} mode...")
        process_crypto(symbol, input_file, output_file, mode=mode)
        
        # Verify data
        df = pd.read_csv(output_file)
        print(f"  Shape: {df.shape}")
        print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

# Cell 5: Configure training
import torch
from src.models.train_enhanced_transformer import EnhancedTransformerTrainer, TradingConfig

# Training configurations for different modes
configs = {
    'regression': {
        'task_type': 'regression',
        'hidden_dim': 256,  # Reduced for GPU efficiency
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.3,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'epochs': 50,
        'patience': 10,
        'window_size': 16
    },
    '3class': {
        'task_type': '3class',
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.3,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'epochs': 50,
        'patience': 10,
        'window_size': 16
    }
}

# Enable mixed precision for faster training
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Cell 6: Train models
import json
from pathlib import Path

# Create output directory
output_dir = Path('/kaggle/working/trained_models')
output_dir.mkdir(exist_ok=True)

training_results = {}

for mode, config_dict in configs.items():
    print(f"\n{'='*50}")
    print(f"Training {mode} model")
    print(f"{'='*50}")
    
    # Create config
    config = TradingConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    # Initialize trainer
    trainer = EnhancedTransformerTrainer(config)
    
    # Train on Bitcoin data (you can change to other symbols)
    data_path = f'data/processed/bitcoinusdt_enriched_{mode}.csv'
    
    # Train model
    history = trainer.train(
        data_path=data_path,
        mode=mode,
        save_dir=str(output_dir / mode)
    )
    
    # Save training results
    training_results[mode] = {
        'config': config_dict,
        'history': history,
        'final_loss': history['val_losses'][-1] if history['val_losses'] else None
    }
    
    print(f"{mode} training completed!")
    print(f"Final validation loss: {training_results[mode]['final_loss']:.4f}")

# Cell 7: Evaluate models
from src.models.evaluate_model import ModelEvaluator

evaluation_results = {}

for mode in configs.keys():
    print(f"\n{'='*50}")
    print(f"Evaluating {mode} model")
    print(f"{'='*50}")
    
    # Load trained model
    model_path = output_dir / mode / f'best_{mode}_model.pt'
    evaluator = ModelEvaluator(str(model_path))
    
    # Evaluate on test data
    data_path = f'data/processed/galausdt_enriched_{mode}.csv'  # Test on different symbol
    metrics = evaluator.evaluate(data_path, save_dir=str(output_dir / f'{mode}_evaluation'))
    
    evaluation_results[mode] = metrics
    
    print(f"\n{mode} evaluation results:")
    print(f"Best strategy: {metrics['best_strategy']}")
    print(f"Best Sharpe ratio: {metrics['best_sharpe']:.3f}")
    print(f"Best total return: {metrics['best_return']:.2%}")

# Cell 8: Compare results
import matplotlib.pyplot as plt
import seaborn as sns

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Sharpe Ratio Comparison
sharpe_ratios = [evaluation_results[mode]['best_sharpe'] for mode in modes]
axes[0, 0].bar(modes, sharpe_ratios, alpha=0.7)
axes[0, 0].set_title('Best Sharpe Ratio by Mode')
axes[0, 0].set_ylabel('Sharpe Ratio')

# Plot 2: Total Return Comparison
total_returns = [evaluation_results[mode]['best_return'] for mode in modes]
axes[0, 1].bar(modes, total_returns, alpha=0.7, color='green')
axes[0, 1].set_title('Best Total Return by Mode')
axes[0, 1].set_ylabel('Total Return')

# Plot 3: Strategy Performance for Best Mode
best_mode = max(modes, key=lambda m: evaluation_results[m]['best_sharpe'])
strategies = list(evaluation_results[best_mode]['backtest_results'].keys())
returns = [evaluation_results[best_mode]['backtest_results'][s]['total_return'] for s in strategies]
drawdowns = [evaluation_results[best_mode]['backtest_results'][s]['max_drawdown'] for s in strategies]

x = np.arange(len(strategies))
width = 0.35
axes[1, 0].bar(x - width/2, returns, width, label='Return', alpha=0.7)
axes[1, 0].bar(x + width/2, np.abs(drawdowns), width, label='Drawdown', alpha=0.7, color='red')
axes[1, 0].set_title(f'Strategy Performance ({best_mode} mode)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(strategies)
axes[1, 0].legend()

# Plot 4: Training Loss Curves
for mode in modes:
    losses = training_results[mode]['history']['val_losses']
    axes[1, 1].plot(losses, label=f'{mode}', alpha=0.7)
axes[1, 1].set_title('Training Loss Curves')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Validation Loss')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(output_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Cell 9: Save comprehensive results
results = {
    'training_results': training_results,
    'evaluation_results': evaluation_results,
    'summary': {
        'best_mode': best_mode,
        'best_sharpe_ratio': evaluation_results[best_mode]['best_sharpe'],
        'best_total_return': evaluation_results[best_mode]['best_return'],
        'best_strategy': evaluation_results[best_mode]['best_strategy']
    },
    'gpu_info': {
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
    }
}

with open(output_dir / 'training_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*50}")
print("TRAINING COMPLETED!")
print(f"{'='*50}")
print(f"Best mode: {best_mode}")
print(f"Best Sharpe ratio: {results['summary']['best_sharpe_ratio']:.3f}")
print(f"Best total return: {results['summary']['best_total_return']:.2%}")
print(f"Best strategy: {results['summary']['best_strategy']}")
print(f"\nAll results saved to: {output_dir}")
print(f"\nFiles to download:")
print(f"- Model weights: {output_dir}/{best_mode}/best_{best_mode}_model.pt")
print(f"- Evaluation plots: {output_dir}/{best_mode}_evaluation/")
print(f"- Comparison plots: {output_dir}/comparison_plots.png")
print(f"- Full results: {output_dir}/training_results.json")

# Cell 10: Optional: Sync results back to GitHub
# Uncomment and modify if you want to automatically push results back

# !git config --global user.email "your-email@example.com"
# !git config --global user.name "Your Name"
# 
# !git add .
# !git commit -m "Add trained models and evaluation results"
# !git push origin main
