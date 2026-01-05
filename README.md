# Enhanced ML Crypto Trading System

A machine learning-based cryptocurrency trading system that uses Transformer architectures with **regression-based targets** and **risk-adjusted performance optimization**. This project demonstrates proper ML engineering practices for financial time series prediction.

## Key Features

- **Regression & 3-Class Classification**: Moved beyond binary classification to predict actual returns or directional movements
- **Sequence Pooling**: Uses mean pooling instead of last-token-only for better temporal representation
- **Signal-Gated Trading**: Trades only when signals exceed thresholds with volatility filters
- **Risk Management**: Built-in position sizing, stop-loss, drawdown limits, and transaction costs
- **Kaggle Integration**: GPU-optimized training notebooks for efficient resource usage
- **Evaluation Focus**: Emphasizes Sharpe ratio and risk-adjusted returns over accuracy

## Why Not Accuracy?

In financial markets, **high accuracy ≠ profitability**. This project focuses on:
- Expected returns and risk-adjusted performance
- Realistic trading costs and slippage
- Signal strength and volatility filtering
- Portfolio-level metrics (Sharpe ratio, drawdown)

## Project Structure

```
├── configs/             # Model and training configurations
├── data/                # Raw and processed cryptocurrency data
├── models/              # Enhanced Transformer models and training scripts
├── src/                 # Source code
│   ├── data/           # Feature engineering and data processing
│   ├── models/         # Model implementations and training
│   └── trading/        # Enhanced backtesting and strategy
├── kaggle_training_notebook.py  # Kaggle GPU training script
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Local Development
```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Process data for regression
python src/data/feature_engineering.py

# Train enhanced model
python src/models/train_enhanced_transformer.py

# Evaluate with risk-adjusted metrics
python src/models/evaluate_model.py
```

### Kaggle GPU Training
1. Upload your project to Kaggle as a dataset
2. Copy `kaggle_training_notebook.py` into a new notebook
3. Enable GPU accelerator
4. Run all cells

## 📊 Model Architecture

### Enhanced Transformer
- **Input**: Multi-feature time series (price, volume, technical indicators)
- **Architecture**: 4-layer Transformer with sequence pooling
- **Output**: Regression (returns) or 3-class classification (down/neutral/up)
- **Training**: MSE/Huber loss for regression, CrossEntropy for classification

### Key Improvements
- **Sequence Pooling**: `x.mean(dim=1)` instead of last token only
- **Reduced Depth**: 4 layers for GPU efficiency
- **Flexible Targets**: Regression or 3-class classification
- **Risk-Aware**: Trading strategy with position sizing and stop-loss

## 📈 Evaluation Metrics

Instead of accuracy, we focus on:
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Directional Accuracy**: Correct sign predictions (for regression)

## ⚙️ Configuration

All configurations are in `configs/`:
- `model_config.yaml`: Model architecture and task settings
- `training_config.yaml`: Environment-specific training parameters

## 🔧 Usage Examples

### Training Different Modes
```python
# Regression mode
config.task_type = 'regression'
trainer.train(data_path, mode='regression')

# 3-class mode
config.task_type = '3class'
trainer.train(data_path, mode='3class')
```

### Backtesting Strategies
```python
# Conservative strategy
config = TradingConfig(min_signal_threshold=0.005, max_volatility_threshold=0.02)

# Aggressive strategy
config = TradingConfig(min_signal_threshold=0.001, max_volatility_threshold=0.05)
```

## 🎯 Project Philosophy

This project demonstrates:
1. **Proper ML Engineering**: Clean code, configs, reproducibility
2. **Financial ML Best Practices**: Risk management, realistic evaluation
3. **Resource Awareness**: Kaggle integration for GPU training
4. **Portfolio Readiness**: Professional documentation and structure

## 📝 License

MIT License - see LICENSE file