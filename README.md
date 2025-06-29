# Trading Bot with Transformer Architecture

A machine learning-based trading bot that uses a Transformer architecture for pattern recognition in cryptocurrency time series data.

## Project Structure

```
├── data/                 # Processed and raw financial data
├── models/              # Trained model checkpoints and configurations
├── src/                 # Main source code
│   ├── data/           # Data processing modules
│   ├── models/         # Model implementation
│   ├── training/       # Training scripts
│   └── trading/        # Trading strategy implementation
└── tests/              # Test files
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

## Features

- Transformer-based architecture for time series analysis
- Direct processing of raw cryptocurrency time series data
- Advanced pattern recognition through self-attention mechanisms
- Real-time trading capabilities
- Multi-cryptocurrency support (Bitcoin and Gala)
- Improved scalability and performance compared to previous CNN-based approach

## Project Development Timeline

### Phase 1: Initial Setup
- Created project structure with src/, data/, and models/ directories
- Set up requirements.txt with necessary dependencies
- Implemented basic data processing pipeline

### Phase 2: Architecture Migration
- Initially attempted to use Inception V3 for pattern recognition
- Switched to Transformer architecture due to:
  - Better suitability for time series data
  - More efficient handling of sequential patterns
  - Direct processing of raw time series data without image conversion

### Phase 3: Transformer Implementation
- Implemented TradingTransformer model with:
  - Custom positional encoding for time series data
  - Multi-head attention mechanism
  - Configurable architecture parameters
  - Input Features (11 total):
    - Open price
    - High price
    - Low price
    - Close price
    - Volume
    - Marketcap/Quote volume
    - RSI (14 periods)
    - MACD
    - MACD Signal
    - MACD Histogram
    - EMA (20 periods)
  - Model Architecture:
    - Input dimension: 11
    - Hidden dimension: 128
    - Number of layers: 6
    - Number of attention heads: 8
    - Dropout: 0.1
    - Window size: 8
- Updated data processing pipeline to handle time series sequences
- Modified training script for binary prediction using BCEWithLogitsLoss

### Phase 4: Current Status
- Working on debugging positional encoding shape issues
- Preparing for training on Bitcoin and Gala datasets
- Planning to implement trading strategy based on Transformer predictions

## Technical Details

### Current Performance

After training with technical indicators:

- **Validation Loss**: ~0.67
- **Accuracy**: ~62.57%
- **AUC**: ~0.48

The model is currently not making any positive predictions (all predictions are negative), indicating a need for further improvements.

## Usage

1. Process data:
```bash
python src/data/process_data.py
```

2. Train model:
```bash
python src/training/train_sample.py
```

3. Start trading bot:
```bash
python src/trading/trading_bot.py
```

## Next Steps

1. Add more technical indicators:
   - Bollinger Bands
   - On-Balance Volume (OBV)
   - Average True Range (ATR)
   - ADX (Average Directional Index)

2. Model Architecture Improvements:
   - Increase hidden dimension
   - Add more layers
   - Adjust dropout rate
   - Add batch normalization

3. Hyperparameter Tuning:
   - Learning rate
   - Batch size
   - Window size
   - Number of layers

The goal is to reach at least 80% accuracy in price movement prediction.

## License

This project is licensed under the MIT License - see the LICENSE file.