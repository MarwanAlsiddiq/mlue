# Project checkpoint

Status recorded: 6 September 2026. Source reviewed at commit `3cf5945`.

## Decision and reason

Active development is paused because of the cost of training and iterating on the model, as stated by the maintainer. The repository is retained as an exploratory ML prototype with a documented stopping point.

No historical spend, GPU-hour total, exhausted quota, training duration or budget figure is asserted: those records were not supplied. A need for more compute is also not evidence that additional training would produce a useful model.

The findings below describe the current engineering state. They are not presented as the original reason for stopping; training cost is that reason.

## Work preserved

The repository includes market-data acquisition and processing, technical indicators, return/direction labels, model experiments, training/checkpoint code, evaluation metrics and simulated strategy logic. It also contains a Kaggle notebook script and environment-specific presets.

Historical checkpoints, processed arrays and a serialized scaler are retained. They were inventoried rather than loaded or retrained during this review. No claim is made that either checkpoint corresponds to the current enhanced Transformer implementation.

The inspectable work connects prediction with downstream evaluation, including transaction-cost and risk-control settings. The effectiveness and correctness of those settings still require validation.

## Readiness boundaries

| Status | Meaning |
| --- | --- |
| IMPLEMENTED | Data/features/targets, models, training loops, baselines and backtesting logic are present |
| HISTORICAL ARTIFACTS | Weights and processed data exist; their complete provenance/evaluation linkage is unverified |
| UNVALIDATED | End-to-end reproduction, leakage-free performance, strategy accounting and model usefulness |
| PAUSED | Further training and experimental iteration because of compute cost |

## Findings before further training

These are source-review findings, not outcomes from new training runs.

| Finding | Evidence and required check |
| --- | --- |
| Scaling precedes validation split | `EnhancedTransformerTrainer.prepare_data()` fits its scaler on all selected rows before splitting. Use training-only preprocessing and verify chronological/horizon separation. |
| Sequence construction is duplicated | `prepare_data()` creates windows, then `TradingDataset` slices windows again. Build sequences once and test batch × time × feature dimensions. |
| Tensor layout needs verification | `forward()` documents batch-first input, while `TransformerEncoderLayer` does not explicitly set `batch_first`. Establish the intended layout and test sample independence. |
| Features/configuration are not one contract | Feature engineering expands columns; model defaults and YAML lists differ; the enhanced trainer constructs a Python config directly. Define and test one ordered feature contract. |
| Execution paths need repair | Some modules use relative imports and hard-coded paths. The Kaggle file contains notebook `!` commands. Separate notebook instructions from Python commands and parameterize paths. |
| Dependencies need reconciliation | `requirements.txt` repeats `torchvision` requirements. Source imports include absent packages such as `ta` and `seaborn`. Establish a tested environment before providing installation instructions. |
| No functional test coverage | Both files under `tests/` are empty. Add alignment, shape and deterministic backtest-accounting tests. |
| Result provenance is incomplete | Historical weights and arrays lack a verified accompanying run manifest. Record data/splits, code revision, seed, resolved config and metrics together. |
| License text is missing | `LICENSE` is empty despite the historical README's MIT claim. Clarify rights and intended licensing before adding license text. |

The backtest also needs independent accounting checks before its Sharpe ratio, drawdown or returns are used as evidence. Unverified artifact metrics are not promoted to the README.

## Conditions for resuming

1. **Correctness first:** add small CPU tests for splits, target alignment, feature order, sequence shape and strategy accounting; repair interfaces before a large training run.
2. **A reproducible baseline:** use the same chronological holdout for a simple baseline and the model, with training-only preprocessing and explicit window/horizon separation.
3. **A bounded pilot:** record model/data size, hardware and measured step time. Define a spending limit before launching paid compute.
4. **A costed experiment:** estimate cost from measured runtime and the chosen provider's actual rate, including planned repeats. No provider price or spend estimate is invented here.
5. **An inspectable result:** retain the run manifest, checkpoint linkage, unsuccessful attempts and held-out evaluation before deciding on larger experiments.

These are resumption criteria, not a scheduled roadmap or a promise to restart. This checkpoint launches no training, cloud resources, API calls or trades.

## Verification of this checkpoint

Reviewed the README, model/training/feature code, configurations, notebook script and artifact inventory. `python -m unittest discover -s tests -v` discovered **zero tests**; that is not a passing functional suite. Documentation links and changed text were checked before publication. Training dependencies were not installed and model performance was not evaluated.

Only documentation and repository presentation are updated. Existing experiments and artifacts remain available; the repository has not been archived or deleted.
