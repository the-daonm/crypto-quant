# Model comparison (design period)

Does a more complex model predict better than the two models the project started with? This page compares
five models and Buy & Hold on the **design period only** (out-of-sample walk-forward folds before 2025-01-01).
The holdout period was not touched: `scripts/compare_models.py` cuts the data at `holdout_start` before fitting
anything.

## Setup

- Pairs: BTC/USDT, ETH/USDT, SOL/USDT on `1d` and `4h` (6 pairs), Binance data via ccxt, downloaded 2026-11-12.
- Default configuration from `config.yaml`: `horizon = 3`, `threshold = 0.5`, `band = 0.05`, `fee = 0.1%`.
- **Configurations tried: one per model.** The hyperparameters below were written down before looking at any
  result and were not tuned (tuning belongs to the nested walk-forward task, issue #5). The only change made
  after the first run was raising the MLP iteration budget from 500 to 2000 because the optimiser had not
  converged; that decision was based on the convergence warning, not on any metric.

| Model | Library | Configuration |
|---|---|---|
| `logreg` | scikit-learn | standardised inputs, `C = 0.1` |
| `lightgbm` | LightGBM | 200 trees, depth 3, learning rate 0.03, 80% row/column subsampling |
| `random_forest` | scikit-learn | 300 trees, depth 4, at least 50 samples per leaf, `sqrt` features per split |
| `xgboost` | XGBoost | 200 trees, depth 3, learning rate 0.03, 80% row/column subsampling (mirrors LightGBM) |
| `mlp` | scikit-learn | standardised inputs, one hidden layer of 16 units, L2 penalty `alpha = 1.0` |

## Results: average over 6 pairs, after fees

| Model | AUC | Accuracy | Sharpe | Max drawdown | Total return | Trades | Training time (s) | Beats B&H Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Buy & Hold | – | – | **0.97** | −72% | +296% | – | – | – |
| `logreg` | 0.518 | 51.3% | 0.69 | −63% | +98% | 198 | 0.3 | 1/6 |
| `lightgbm` | 0.529 | 52.1% | 0.87 | **−59%** | +205% | 345 | 1.1 | 3/6 |
| `random_forest` | 0.529 | 52.0% | 0.72 | −67% | +106% | 27 | 9.9 | 1/6 |
| `xgboost` | **0.533** | **52.3%** | 0.83 | −62% | +164% | 346 | 3.2 | 2/6 |
| `mlp` | 0.524 | 51.9% | 0.75 | −65% | +159% | 308 | 16.3 | 2/6 |

Training time is the sum of all walk-forward fits for one pair, averaged over the pairs, on a laptop CPU.
It is indicative only.

### Sharpe ratio per pair

| Pair | Buy & Hold | `logreg` | `lightgbm` | `random_forest` | `xgboost` | `mlp` |
|---|---:|---:|---:|---:|---:|---:|
| BTC 1d | 0.83 | 0.86 | 0.46 | 0.66 | 0.64 | 0.19 |
| BTC 4h | 0.71 | 0.63 | 1.06 | 0.74 | 0.90 | 1.01 |
| ETH 1d | 0.57 | 0.38 | 0.10 | 0.35 | 0.07 | 0.10 |
| ETH 4h | 0.64 | 0.17 | 0.75 | 0.49 | 0.47 | 0.19 |
| SOL 1d | 1.78 | 1.49 | 1.55 | 1.48 | 1.79 | 1.67 |
| SOL 4h | 1.29 | 0.60 | 1.32 | 0.59 | 1.10 | 1.33 |

## Commentary

1. **No model stands out.** The average AUC of every model lies between 0.52 and 0.53. The spread between models
   (about 0.015) is smaller than the spread of a single model across pairs (LightGBM ranges from 0.49 to 0.55),
   so the ranking is not reliable evidence that one model is better.
2. **More complexity did not help.** The neural network is the slowest model (about 15× LightGBM) and is not more
   accurate. With roughly 2,000 daily bars and a signal this weak, there is little for a flexible model to learn;
   this is the evidence behind not going further into deep learning (LSTM) for now.
3. **XGBoost behaves like LightGBM**, as expected from near-identical configurations: similar AUC, trade count and
   drawdown. It is about 3× slower here and adds nothing LightGBM does not already provide.
4. **Random Forest barely trades** (27 trades versus about 345). Averaging 300 shallow trees gives probabilities
   very close to 0.5, which rarely leave the ±0.05 hysteresis band. Its trading results therefore say more about
   the band than about the model; a fair comparison would need a model-specific band, which is a tuning question.
5. **Nothing beats Buy & Hold on average Sharpe in the design period**, consistent with the README: the period was
   a bull market and the value of the strategy is lower drawdown, not higher return.

The default model stays **LightGBM**: it is the fastest of the tree ensembles and no alternative is clearly better.
These numbers come from one run per model, without significance tests (see issue #6).

## Reproduce

```bash
pip install -e ".[dev]"
python scripts/compare_models.py     # prints the tables and writes results/model_comparison*.csv
```
