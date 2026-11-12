# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- MIT License, contributing guide (`CONTRIBUTING.md`) and Code of Conduct (Contributor Covenant 2.1).
- Issue templates (bug report, feature request, experiment) and a pull request template.
- Project metadata in `pyproject.toml` (license, URLs, keywords, classifiers).
- Linting and formatting with `ruff`.

### Changed
- Code, dashboard, notebook and project documentation are now in English;
  the beginner explainer `docs/GIAI-THICH.md` remains in Vietnamese.

## [0.2.0] - 2026-11-12

### Added
- Multi-bar labels (`horizon`) with a matching embargo in walk-forward validation.
- Hysteresis band (`band`) on entry/exit signals to reduce trading.
- Separate design and holdout (`holdout_start`) reporting in the CLI, dashboard and notebook.
- `docs/GIAI-THICH.md`: beginner explainer (Vietnamese).

### Changed
- Default configuration is now `horizon = 3`, `band = 0.05`. On the 2025–2026 holdout, the average LightGBM
  Sharpe ratio improved from −1.28 to +0.10.

## [0.1.0] - 2026-11-12

### Added
- OHLCV download from Binance via ccxt with a parquet cache.
- 21 causal technical features and next-bar direction labels.
- Baseline, Logistic Regression and LightGBM models; walk-forward validation; `auto` threshold selection.
- Fee-aware long/flat backtest; classification and trading metrics.
- `python -m cq.pipeline` CLI, Streamlit dashboard and EDA notebook.

[Unreleased]: https://github.com/the-daonm/crypto-quant/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/the-daonm/crypto-quant/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/the-daonm/crypto-quant/releases/tag/v0.1.0
