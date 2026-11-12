# Changelog

Mọi thay đổi đáng kể của dự án được ghi ở đây.
Định dạng theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/), phiên bản theo [Semantic Versioning](https://semver.org/lang/vi/).

## [Unreleased]

### Added
- MIT License, hướng dẫn đóng góp (`CONTRIBUTING.md`), quy tắc ứng xử (`CODE_OF_CONDUCT.md`).
- Template cho issue (báo lỗi, đề xuất, thí nghiệm) và pull request.
- Metadata dự án trong `pyproject.toml` (license, URL, keywords).
- Lint và format bằng `ruff`.

## [0.2.0] - 2026-11-12

### Added
- Nhãn nhìn nhiều nến (`horizon`), kèm embargo tương ứng trong walk-forward.
- Tín hiệu có vùng đệm (hysteresis, `band`) để giảm số lệnh.
- Báo cáo tách riêng giai đoạn design và holdout (`holdout_start`) trong CLI, dashboard và notebook.
- `docs/GIAI-THICH.md`: tài liệu giải thích kiến thức cho người mới.

### Changed
- Cấu hình mặc định: `horizon = 3`, `band = 0.05`. Trên holdout 2025–2026, Sharpe trung bình của LightGBM
  tăng từ -1.28 lên +0.10.

## [0.1.0] - 2026-11-12

### Added
- Tải dữ liệu OHLCV từ Binance qua ccxt, có cache parquet.
- 21 feature kỹ thuật causal, nhãn hướng giá nến kế tiếp.
- Mô hình baseline, Logistic Regression, LightGBM; walk-forward validation; ngưỡng `auto`.
- Backtest long/flat có phí; metric phân loại và metric giao dịch.
- CLI `python -m cq.pipeline`, dashboard Streamlit, notebook EDA.

[Unreleased]: https://github.com/the-daonm/crypto-quant/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/the-daonm/crypto-quant/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/the-daonm/crypto-quant/releases/tag/v0.1.0
