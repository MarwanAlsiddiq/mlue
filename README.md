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

---

## Progress Update: July 2025

### Major Work Done
- **Fetched extensive multi-timeframe historical data** (1m, 5m, 15m, 30m, 1h, 4h) for BTCUSDT and GALAUSDT from Binance.
- **Engineered a wide set of technical indicators** using the `ta` library (EMA, SMA, Bollinger Bands, ATR, RSI, MACD, OBV, CCI, Williams %R, Stochastic RSI, ADX, MFI, lagged returns, rolling volatility, time features).
- **Addressed class imbalance** with RandomOverSampler for LightGBM and dynamic class weighting for deep learning.
- **Trained and evaluated LightGBM models** (macro F1/accuracy low, models often predicted only majority class).
- **Trained and evaluated a Transformer model** on enriched 15m BTCUSDT data, with and without class weighting:
  - Initial accuracy ~61%, but model predicted only one class (no positives).
  - Lowered label threshold from 0.2% to 0.05% to increase positive samples (now ~40% positive labels).
  - Retrained Transformer with class weighting (pos_weight=1.48): accuracy ~63%, but still no positive predictions.

### Current Status
- **Data and features are robust and multi-scale.**
- **Models (LightGBM, Transformer) still predict only majority class, even after aggressive class weighting and label threshold lowering.**
- **Next steps:**
  - Debug model outputs (logits/probabilities) to check if the model is ever close to predicting positives.
  - Try even lower thresholds, different features, or alternate modeling approaches.

---

## Project Journey: Steps, Challenges, and Solutions

### 1. Initial Goal and Model
- Objective: Achieve >80% accuracy in predicting significant upward price moves for cryptocurrencies.
- Started with a Transformer-based sequence model using minute-level crypto data (BTC, GALA).
- Initial results: Accuracy stuck near 50%, precision/recall/F1 very low, model predicted almost all negatives.

### 2. Diagnosing Issues
- **Label Noise:** Binary labels (next close > last close) were very noisy and nearly random at minute-level granularity.
- **Class Imbalance:** Significant upward moves were rare, leading to severe imbalance and misleading accuracy.
- **Data Quality:** No data leakage, but label distribution was highly skewed for some coins.

### 3. Solutions and Improvements
- **Feature Engineering:**
  - Added technical indicators: EMA, SMA, Bollinger Bands, ATR, RSI, MACD, OBV, lagged returns, rolling volatility, time features.
  - Used the `ta` library for robust indicator calculation.
- **Label Redefinition:**
  - Switched to a 'significant move' label: 1 if next close > last close by 0.5% or more, else 0.
- **Class Imbalance Handling:**
  - Used RandomOverSampler to balance classes in the training set for all models.
- **Aggregation to Higher Timeframes:**
  - Aggregated data to 15m, 30m, 1h, and 4h intervals to reduce noise and increase the predictability of significant moves.
  - Automated aggregation and feature engineering for all coins and all timeframes.
- **Modeling:**
  - Used LightGBM for tabular modeling with oversampling.
  - Evaluated with accuracy, F1, precision, recall, and confusion matrix for every coin/timeframe.

### 4. Key Challenges and How We Overcame Them
- **Label Noise and Randomness:**
  - Initial binary labels were too noisy; switching to threshold-based labels improved learnability.
- **Class Imbalance:**
  - Oversampling helped, but rare positives still limited recall.
- **Data Sparsity at High Timeframes:**
  - 1h/4h aggregations resulted in very few samples; focused on 15m/30m/1h for more reliable modeling.
- **Model Predictive Power:**
  - Even after improvements, models often predicted all negatives due to label rarity and data limits.
  - High accuracy was often due to class imbalance, not real predictive skill (low F1/recall).

### 5. Current Limitations and Next Steps
- **Significant-move prediction remains very challenging** due to label rarity and data limitations.
- **No coin/timeframe achieved both high accuracy and meaningful F1/recall**—further work is needed on label definition, data volume, or alternative modeling approaches (e.g., regression, multi-class, or higher-level signals).
- **Next steps:**
  - Consider lowering the threshold for significant moves to increase positive samples.
  - Try alternative targets (multi-step, regime prediction, regression).
  - Expand data sources (order book, sentiment, etc.) if available.
  - Focus on timeframes/coins with more data for deep learning.

---

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