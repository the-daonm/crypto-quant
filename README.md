# crypto-quant

Dự đoán **hướng giá** (tăng/giảm) của nến kế tiếp cho BTC, ETH, SOL bằng ML cổ điển, sau đó backtest
chiến lược long/flat **có tính phí** bằng walk-forward validation và so sánh với Buy & Hold.

> Project học tập, không phải lời khuyên đầu tư.

## Cài đặt và chạy

```bash
python -m venv .venv && source .venv/bin/activate     # fish: source .venv/bin/activate.fish
pip install -e ".[dev,notebook]"

pytest                                                 # test chống leakage và kiểm tra backtest
python -m cq.pipeline --symbol BTC/USDT --tf 1d        # in bảng metrics, lưu CSV vào results/
python -m cq.pipeline --symbol ETH/USDT --tf 4h --threshold auto --fee 0.0005
streamlit run app.py                                   # dashboard
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
| Feature (`features.py`) | 24 feature kỹ thuật: lag return, momentum, volatility, RSI, MACD hist, Bollinger %B, ATR, khoảng cách tới MA20/50/200, volume z-score, … Tất cả đều **causal** (chỉ dùng dữ liệu ≤ t). |
| Nhãn (`labels.py`) | `y_t = 1` nếu `close_{t+1} > close_t` |
| Mô hình (`models.py`) | Baseline luôn long · Logistic Regression (chuẩn hoá, C=0.1) · LightGBM (nông, regularize mạnh) |
| Validation (`validation.py`) | **Walk-forward expanding window**: train tối thiểu khoảng 2 năm, test khoảng 3 tháng, embargo 1 nến, fit lại model ở mỗi fold. Tất cả kết quả đều là out-of-sample. |
| Ngưỡng | Cố định (0.5) hoặc `auto`: chọn trên 25% cuối của tập **train** theo Sharpe, không bao giờ nhìn tập test |
| Backtest (`backtest.py`) | Tín hiệu tại close nến t, giữ vị thế tới close nến t+1; phí 0.1% mỗi lần đổi vị thế; chỉ long/flat |
| Metric (`metrics.py`) | Accuracy, AUC, F1 · Tổng lợi nhuận, CAGR, Sharpe, Max Drawdown, exposure, win rate, số lệnh |

Kiểm tra trong `tests/`:
- Thay đổi dữ liệu tương lai thì feature quá khứ không được thay đổi.
- Luôn long với phí 0 phải cho kết quả đúng bằng Buy & Hold.
- Một lần vào lệnh bị trừ phí đúng một lần; một vòng mua-bán bị trừ phí hai lần.
- Các fold walk-forward không chồng lấn và có embargo.

## Kết quả (out-of-sample, phí 0.1%, `--threshold auto`, dữ liệu tới 11/2026)

**Khung 1d**

| Coin | Mô hình | Tổng LN | Sharpe | Max DD | Accuracy | AUC | Số lệnh |
|---|---|---:|---:|---:|---:|---:|---:|
| BTC | Buy & Hold | +140% | 0.58 | -77% | – | – | – |
| BTC | LogReg | +28% | 0.30 | **-37%** | 51.4% | 0.520 | 354 |
| BTC | LightGBM | -34% | -0.01 | -69% | 52.0% | 0.523 | 658 |
| ETH | Buy & Hold | +13% | 0.38 | -79% | – | – | – |
| ETH | LogReg | -14% | 0.13 | -63% | 51.1% | 0.524 | 424 |
| ETH | LightGBM | **+110%** | **0.54** | -70% | 50.8% | 0.512 | 528 |
| SOL | Buy & Hold | +267% | 0.83 | -76% | – | – | – |
| SOL | LogReg | +179% | 0.79 | **-42%** | 50.1% | 0.493 | 256 |
| SOL | LightGBM | +34% | 0.43 | -82% | 52.2% | 0.517 | 420 |

**Khung 4h: tác động của phí (BTC, LightGBM, ngưỡng 0.5, khoảng 5 950 lệnh)**

| Phí mỗi lệnh | 0% | 0.02% | 0.05% | 0.10% | Buy & Hold |
|---|---:|---:|---:|---:|---:|
| Tổng lợi nhuận | +1765% | +467% | -5% | -95% | +103% |
| Sharpe | 1.40 | 0.92 | 0.19 | -1.02 | 0.50 |

### Nhận xét

1. **Tín hiệu có nhưng rất yếu.** Accuracy out-of-sample chỉ khoảng 50–55%, AUC khoảng 0.51–0.56. Tự tương quan
   của return gần bằng 0; chỉ volatility là có tính "nhớ" rõ (volatility clustering, xem notebook).
2. **Phí quyết định tất cả.** Ở khung 4h, mô hình có accuracy khoảng 54–55% và có lãi rất lớn nếu không có phí,
   nhưng với phí 0.1% thì hàng nghìn lệnh làm mất gần hết vốn.
3. **Ở khung 1d, không mô hình nào thắng Buy & Hold một cách ổn định** trên cả 3 coin. LightGBM thắng trên
   ETH nhưng thua trên BTC và SOL. LogReg ít giao dịch hơn, **giảm drawdown khoảng một nửa** nhưng đổi lại lợi nhuận thấp hơn.
4. Kết quả thay đổi nhiều theo coin và theo ngưỡng, nên không nên tin vào một cấu hình "tốt nhất" duy nhất
   (rủi ro overfit khi chọn cấu hình).

### Hướng phát triển
- Giảm số lệnh: giữ vị thế tối thiểu N nến, dùng vùng hysteresis cho ngưỡng, đưa phí vào hàm mục tiêu
- Position sizing theo volatility, thêm short
- Dữ liệu ngoài giá: funding rate, open interest, on-chain
- Tune hyperparameter bằng nested walk-forward (Optuna)

## Cấu trúc

```
src/cq/        data · features · labels · models · validation · backtest · metrics · pipeline
app.py         Streamlit dashboard
notebooks/     EDA và phân tích kết quả
tests/         pytest
config.yaml    symbols, timeframe, phí, cửa sổ walk-forward
```
