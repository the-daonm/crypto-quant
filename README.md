# crypto-quant

Dự đoán **hướng giá** (tăng/giảm) của BTC, ETH, SOL bằng ML cổ điển, sau đó backtest chiến lược
long/flat **có tính phí** bằng walk-forward validation và so sánh với Buy & Hold.

> Project học tập, không phải lời khuyên đầu tư.
> Mới làm quen với crypto, ML hay backtest? Đọc **[docs/GIAI-THICH.md](docs/GIAI-THICH.md)** trước:
> file này giải thích từ đầu tất cả kiến thức cần thiết và từng phần của project.

## Cài đặt và chạy

```bash
python -m venv .venv && source .venv/bin/activate     # fish: source .venv/bin/activate.fish
pip install -e ".[dev,notebook]"

pytest                                                 # 13 test: chống leakage, backtest, walk-forward
python -m cq.pipeline --symbol BTC/USDT --tf 1d        # metrics toàn giai đoạn + design/holdout, CSV vào results/
python -m cq.pipeline --symbol ETH/USDT --tf 4h --horizon 1 --band 0   # chạy lại cấu hình v1 ban đầu
streamlit run app.py                                   # dashboard: http://localhost:8501
jupyter notebook notebooks/eda.ipynb                   # EDA và phân tích kết quả
```

Lần chạy đầu tiên sẽ tải dữ liệu và cache vào `data/` (không commit). Nếu Binance bị chặn ở mạng của bạn,
đổi `exchange` trong `config.yaml` sang `okx` hoặc `bybit`.

## Dữ liệu

OHLCV lấy từ **Binance public market-data API** qua thư viện [ccxt](https://github.com/ccxt/ccxt),
không cần API key, giai đoạn từ 2019-01-01 (SOL từ 2020-08) đến nay, khung `1d` và `4h`.
Repo không chứa dữ liệu thô; dữ liệu chỉ dùng cho mục đích học tập. Nến chưa đóng cửa sẽ bị loại bỏ.

## Phương pháp

| Bước | Cách làm |
|---|---|
| Feature (`features.py`) | 21 feature kỹ thuật: lag return, momentum, volatility, RSI, MACD hist, Bollinger %B, ATR, khoảng cách tới MA20/50/200, volume z-score, … Tất cả đều **causal** (chỉ dùng dữ liệu ≤ t). |
| Nhãn (`labels.py`) | `y_t = 1` nếu `close_{t+h} > close_t`, mặc định `h = 3` nến |
| Mô hình (`models.py`) | Baseline luôn long · Logistic Regression (chuẩn hoá, C=0.1) · LightGBM (nông, regularize mạnh) |
| Validation (`validation.py`) | **Walk-forward expanding window**: train tối thiểu khoảng 2 năm, test khoảng 3 tháng, embargo `h` nến (vì nhãn chồng lấn), fit lại model ở mỗi fold. Tất cả kết quả đều là out-of-sample. |
| Tín hiệu (`backtest.positions`) | **Hysteresis**: vào lệnh khi p > 0.55, thoát khi p < 0.45, ở giữa thì giữ vị thế cũ. Ngưỡng có thể chọn `auto` (chọn trên tập train) |
| Backtest (`backtest.py`) | Tín hiệu tại close nến t, giữ vị thế tới close nến t+1; phí 0.1% mỗi lần đổi vị thế; chỉ long/flat |
| Design / holdout | Cấu hình được chọn **chỉ trên dữ liệu trước 2025-01-01**; giai đoạn 2025 → nay là holdout và chỉ được xem một lần |
| Metric (`metrics.py`) | Accuracy, AUC, F1 · Tổng lợi nhuận, CAGR, Sharpe, Max Drawdown, exposure, win rate, số lệnh |

Kiểm tra trong `tests/`:
- Thay đổi dữ liệu tương lai thì feature quá khứ không được thay đổi.
- Nhãn nhiều nến được căn đúng, còn backtest vẫn dùng return của 1 nến.
- Luôn long với phí 0 phải cho kết quả đúng bằng Buy & Hold; phí bị trừ đúng số lần đổi vị thế.
- Hysteresis giữ vị thế khi p nằm trong vùng đệm và làm giảm số lệnh.
- Các fold walk-forward không chồng lấn và có embargo.

## Kết quả (out-of-sample, phí 0.1%, dữ liệu tới 11/2026)

**v1** = phiên bản đầu tiên (nhãn 1 nến, ngưỡng 0.5, không có vùng đệm).
**v2** = phiên bản hiện tại (nhãn 3 nến, vùng đệm ±0.05).

### Trung bình trên 6 cặp (3 coin × 2 khung thời gian)

| | Design 2021–2024 Sharpe | Design Max DD | **Holdout 2025–2026 Sharpe** | Holdout Max DD | Thắng B&H trên holdout | Số lệnh (toàn kỳ) |
|---|---:|---:|---:|---:|---:|---:|
| Buy & Hold | **0.97** | -72% | -0.11 | -66% | – | – |
| LightGBM v1 | 0.28 | -70% | -1.28 | -75% | 1/6 | ~1 760 |
| LogReg v2 | 0.68 | -63% | **0.10** | **-51%** | **4/6** | ~200 |
| **LightGBM v2** | 0.87 | **-59%** | **0.10** | **-51%** | **4/6** | ~340 |

### Holdout 2025-01 → 2026-11 từng cặp (LightGBM v2 so với Buy & Hold)

| Cặp | B&H lợi nhuận | v2 lợi nhuận | B&H Sharpe | v2 Sharpe | B&H Max DD | v2 Max DD |
|---|---:|---:|---:|---:|---:|---:|
| BTC 1d | -20% | -11% | -0.07 | -0.00 | -53% | -54% |
| BTC 4h | -21% | -20% | -0.08 | -0.12 | -53% | -48% |
| ETH 1d | -36% | **+49%** | -0.02 | 0.66 | -68% | -44% |
| ETH 4h | -38% | **+32%** | -0.07 | 0.54 | -68% | -54% |
| SOL 1d | -55% | -47% | -0.19 | -0.34 | -76% | -65% |
| SOL 4h | -55% | -24% | -0.22 | -0.15 | -78% | -41% |

### Phí giao dịch: BTC 4h, LightGBM

| Phí mỗi lệnh | 0% | 0.02% | 0.05% | 0.10% |
|---|---:|---:|---:|---:|
| v1: lợi nhuận (5 955 lệnh) | +1765% | +467% | -5% | **-95%** |
| v2: lợi nhuận (907 lệnh) | +903% | +736% | +537% | **+305%** |

### Nhận xét

1. **Tín hiệu có nhưng yếu**: accuracy out-of-sample khoảng 50–55%, AUC khoảng 0.50–0.56.
2. **Phí là yếu tố quyết định.** v1 có tín hiệu nhưng hàng nghìn lệnh làm mất gần hết vốn. v2 giảm số lệnh
   khoảng 5–10 lần nhờ nhãn bớt nhiễu và vùng đệm, nên Sharpe trung bình trên giai đoạn design tăng từ 0.28 lên 0.87.
3. **Cải tiến vẫn giữ được trên holdout** (dữ liệu không dùng khi chọn cấu hình): Sharpe -1.28 → +0.10,
   thắng Buy & Hold ở 4/6 cặp, drawdown trung bình -51% so với -66%.
4. **Trong giai đoạn tăng mạnh (design), v2 vẫn chưa thắng Buy & Hold một cách ổn định.** Giá trị chính
   của mô hình là *giảm rủi ro khi thị trường giảm*, không phải lợi nhuận cao hơn.
5. Hạn chế: holdout chỉ khoảng 22 tháng; 3 coin tương quan mạnh với nhau; khi thí nghiệm có thử khoảng 150 tổ hợp
   cấu hình (trên design) nên vẫn có rủi ro chọn được cấu hình tốt do may mắn. Chưa tính slippage.

### Hướng phát triển
- Position sizing theo volatility, thêm short
- Dữ liệu ngoài giá: funding rate, open interest, on-chain
- Thêm coin hoặc thị trường để kiểm tra độ bền; tune hyperparameter bằng nested walk-forward

## Cấu trúc

```
src/cq/        data · features · labels · models · validation · backtest · metrics · pipeline
app.py         Streamlit dashboard
notebooks/     EDA và phân tích kết quả
docs/          GIAI-THICH.md: giải thích kiến thức và project cho người mới
tests/         pytest
config.yaml    symbols, timeframe, phí, horizon, band, holdout, cửa sổ walk-forward
```
