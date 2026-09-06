# Mlue — cryptocurrency time-series ML research

**Status: research prototype · active development paused because of model-training costs.**

Mlue explores predicting cryptocurrency returns and price direction from market time series, then evaluating those predictions through a simulated trading strategy. The repository contains feature engineering, Transformer models, baseline experiments and backtesting code.

## Why I paused

I paused development because of the cost of training and iterating on the model. Continuing the experiments would mean funding additional compute for training, comparing configurations and validating results. I decided to preserve the prototype at this stage instead of committing to that additional spending.

This is a checkpoint in the research, not a completed or validated trading product. The pause does not establish that the approach works or fails; that would require further evaluation. I have kept the code and historical artifacts available, and documented what would need to be checked before resuming.

## What I worked on

| Area | Implementation available for inspection |
| --- | --- |
| Market-data preparation | Fetching, timeframe aggregation and technical indicators in [src/data](src/data) |
| Prediction targets | Future log returns, three-way direction labels and a legacy binary mode in [feature_engineering.py](src/data/feature_engineering.py) |
| Sequence models | Transformer encoder, positional encoding, mean pooling and task-specific output heads in [transformer_model.py](src/models/transformer_model.py) |
| Training experiments | Training loops, early stopping and checkpoint saving in [train_enhanced_transformer.py](src/models/train_enhanced_transformer.py) |
| Baseline comparisons | [Logistic regression](src/models/logistic_regression_baseline.py), LightGBM and walk-forward experiment scripts in [src/models](src/models) |
| Evaluation | Prediction metrics and a simulated backtester with signal filters, position sizing and configurable transaction costs in [enhanced_backtest.py](src/models/enhanced_backtest.py) |
| Compute experiments | [Kaggle notebook script](notebooks/kaggle_training_notebook.py) and local/GPU configuration presets in [configs](configs) |

These are implemented components, not a claim that every path works together end to end. The prototype mixes several experiments; [the engineering checkpoint](docs/project-status.md) records the integration and validation gaps.

## What the evidence supports

- Source code demonstrates the proposed data → model → prediction → simulated strategy workflow.
- Two historical checkpoint files are present: `models/plutus_best.pt` and `models/enhanced_plutus/best_model.pt`.
- Processed arrays and a scaler are present under `data/plutus_processed/`.
- Those artifacts lack a verified experiment lineage linking a particular data split, configuration, checkpoint and independent evaluation report. Their presence is not a performance result.

**No validated out-of-sample performance, profitability or live-trading result is claimed here.** The historical test files are empty; this repository currently has no functional test coverage. No new model was trained for this documentation checkpoint.

## Review the project without training

Start with the implementation links above and [project status](docs/project-status.md). No GPU, API account or paid service is needed to read the code.

```text
configs/       Experimental model and environment presets
src/data/      Data preparation, features and labels
src/models/    Models, training, baselines and evaluation
src/trading/   Additional trading prototype code
notebooks/     Kaggle-oriented notebook script
models/        Historical checkpoint files
data/         Data folders and historical processed artifacts
outputs/       Output conventions; not a verified results report
tests/         Empty historical placeholders
```

The old quick-start commands have been removed because they implied a reproducible training path that has not been established. Dependencies, paths, feature dimensions and sequence construction need reconciliation first. The Kaggle file contains notebook shell commands and is not an ordinary executable Python script. See [resumption conditions](docs/project-status.md#conditions-for-resuming) before launching training.

## What I would do next

First repair and test the data/model interfaces without a large training run. Then establish a small chronological baseline and profile a bounded pilot before allocating a training budget. Larger experiments should follow only when the pipeline is valid and the compute cost is understood.

The next milestone is a reproducible, budgeted experiment—not simply a larger model.

## License status

The repository's [LICENSE](LICENSE) file is currently empty. The previous README described it as MIT, but that text was not present in the license file. Licensing needs to be clarified; this documentation update does not introduce a new license grant.
