# Giải thích từ A tới Z: crypto, machine learning, backtest và project này

File này dành cho người **chưa biết gì** về tiền ảo, machine learning (ML) hay trading. Đọc lần lượt từ trên xuống,
mỗi phần dùng kiến thức của các phần trước. Cuối file có **bảng thuật ngữ** và **câu hỏi bảo vệ đồ án hay gặp**.

**Mục lục**

1. [Bức tranh tổng quát: project này làm gì?](#1-bức-tranh-tổng-quát-project-này-làm-gì)
2. [Tiền ảo (crypto) cơ bản](#2-tiền-ảo-crypto-cơ-bản)
3. [Dữ liệu giá: nến, OHLCV, return](#3-dữ-liệu-giá-nến-ohlcv-return)
4. [Giao dịch: long, flat, phí, Buy & Hold](#4-giao-dịch-long-flat-phí-buy--hold)
5. [Chỉ báo kỹ thuật (feature)](#5-chỉ-báo-kỹ-thuật-feature)
6. [Machine learning cơ bản](#6-machine-learning-cơ-bản)
7. [Hai mô hình dùng trong project](#7-hai-mô-hình-dùng-trong-project)
8. [Đánh giá mô hình phân loại: accuracy, AUC, F1](#8-đánh-giá-mô-hình-phân-loại-accuracy-auc-f1)
9. [Cái bẫy lớn nhất: dữ liệu chuỗi thời gian và leakage](#9-cái-bẫy-lớn-nhất-dữ-liệu-chuỗi-thời-gian-và-leakage)
10. [Backtest và các chỉ số hiệu quả](#10-backtest-và-các-chỉ-số-hiệu-quả)
11. [Overfitting và cách đánh giá trung thực](#11-overfitting-và-cách-đánh-giá-trung-thực)
12. [Đi qua từng file của project](#12-đi-qua-từng-file-của-project)
13. [Từ v1 tới v2: project đã được cải thiện thế nào](#13-từ-v1-tới-v2-project-đã-được-cải-thiện-thế-nào)
14. [Đọc kết quả và kết luận](#14-đọc-kết-quả-và-kết-luận)
15. [Dùng dashboard](#15-dùng-dashboard)
16. [Câu hỏi bảo vệ đồ án hay gặp](#16-câu-hỏi-bảo-vệ-đồ-án-hay-gặp)
17. [Bảng thuật ngữ](#17-bảng-thuật-ngữ)
18. [Đọc thêm](#18-đọc-thêm)

---

## 1. Bức tranh tổng quát: project này làm gì?

Một câu: **dùng dữ liệu giá quá khứ để đoán giá coin sẽ tăng hay giảm trong vài cây nến tới, rồi kiểm tra
xem nếu giao dịch theo dự đoán đó thì có lãi hơn so với việc mua và giữ hay không.**

Luồng xử lý:

```
 Dữ liệu giá từ sàn         Tính chỉ báo           Mô hình ML             Quy tắc giao dịch         Đánh giá
 (BTC, ETH, SOL)   ──►   (RSI, MACD, ...)  ──►  dự đoán xác suất   ──►  p > 0.55 → mua      ──►  lợi nhuận, rủi ro
  data.py                 features.py           giá tăng              p < 0.45 → bán           so với Buy & Hold
                          labels.py             models.py             backtest.py              metrics.py
                                                validation.py
```

Có ba điều quan trọng cần hiểu ngay từ đầu:

- Project **không** dự đoán giá cụ thể (kiểu "ngày mai BTC = 98 000 USD"). Nó chỉ dự đoán **hướng** (tăng hay giảm),
  vì đây là bài toán dễ đánh giá hơn và có ý nghĩa cho giao dịch.
- Kết quả "đẹp" không phải là mục tiêu. Mục tiêu là **đánh giá đúng phương pháp**: không gian lận dữ liệu,
  có tính phí, có so sánh với phương án đơn giản.
- Dự đoán thị trường tài chính là **cực khó**. Một mô hình đoán đúng 52–55% đã là có tín hiệu. Đừng ngạc nhiên
  khi thấy những con số nhỏ như vậy.

---

## 2. Tiền ảo (crypto) cơ bản

### 2.1 Tiền ảo là gì?

**Tiền ảo / tiền mã hoá (cryptocurrency)** là tài sản số chỉ tồn tại dưới dạng dữ liệu, được ghi nhận trên một
**blockchain**. Blockchain là một cuốn sổ cái công khai, được rất nhiều máy tính trên thế giới cùng lưu giữ, nên
không ai một mình sửa được.

| Coin | Ký hiệu | Đặc điểm |
|---|---|---|
| Bitcoin | **BTC** | Đồng đầu tiên (2009), lớn nhất, thường được coi như "vàng số" |
| Ethereum | **ETH** | Nền tảng chạy chương trình (smart contract), lớn thứ 2 |
| Solana | **SOL** | Blockchain mới hơn, nhanh, biến động mạnh hơn BTC/ETH |

### 2.2 Stablecoin và cặp giao dịch

**USDT (Tether)** là một **stablecoin**: một coin được thiết kế để luôn có giá khoảng 1 USD. Trên sàn, người ta
mua bán theo **cặp giao dịch**:

- **BTC/USDT** = giá 1 BTC tính bằng USDT. Nếu BTC/USDT = 95 000, nghĩa là 1 BTC đổi được 95 000 USDT (khoảng 95 000 USD).
- Mua BTC/USDT = đưa USDT, nhận BTC. Bán = ngược lại.

### 2.3 Sàn giao dịch (exchange)

**Sàn** (Binance, OKX, Bybit, …) là nơi người mua và người bán gặp nhau. Sàn ghi lại mọi giao dịch và công bố
**dữ liệu thị trường** (giá, khối lượng) qua **API**: một địa chỉ web mà chương trình có thể gọi để lấy dữ liệu tự động.

Project dùng **public API** của Binance, tức phần dữ liệu công khai, không cần tài khoản hay key. Thư viện **ccxt**
giúp gọi API của hơn 100 sàn với cùng một cách viết code.

### 2.4 Vì sao chọn crypto cho đồ án?

- Dữ liệu **miễn phí**, dễ lấy, có lịch sử dài.
- Giao dịch **24/7**, không có ngày nghỉ, nên dữ liệu liên tục, ít lỗ hổng.
- Biến động lớn nên thấy rõ các hiện tượng như drawdown hay volatility.

Đổi lại, crypto **rủi ro rất cao**: BTC từng giảm khoảng 77% từ đỉnh 2021 tới đáy 2022, và các coin nhỏ còn giảm mạnh hơn.

---

## 3. Dữ liệu giá: nến, OHLCV, return

### 3.1 Cây nến (candlestick) và OHLCV

Giá thay đổi liên tục từng giây. Để dễ xử lý, người ta gom lại theo từng khoảng thời gian gọi là **nến (candle / bar)**.
Mỗi nến có 5 con số, gọi là **OHLCV**:

| Chữ | Tên | Ý nghĩa |
|---|---|---|
| **O** | Open | Giá đầu khoảng thời gian |
| **H** | High | Giá cao nhất trong khoảng |
| **L** | Low | Giá thấp nhất trong khoảng |
| **C** | Close | Giá cuối khoảng (quan trọng nhất) |
| **V** | Volume | Khối lượng giao dịch trong khoảng |

```
   High ─┬─          Nến xanh: Close > Open (giá tăng)
         │           Nến đỏ:   Close < Open (giá giảm)
  Close ┌┴┐
        │ │  ← thân nến
   Open └┬┘
         │
    Low ─┴─
```

**Khung thời gian (timeframe)**: `1d` = mỗi nến là 1 ngày; `4h` = mỗi nến 4 giờ (mỗi ngày có 6 nến).
Project dùng cả hai: 1d có khoảng 2 900 nến từ 2019, còn 4h có khoảng 17 000 nến.

Lưu ý: crypto dùng giờ **UTC**. Nến 1d bắt đầu lúc 00:00 UTC, tức 07:00 sáng giờ Việt Nam.

### 3.2 Return (lợi suất)

Thay vì nhìn giá tuyệt đối (95 000 hay 30 000), ta nhìn **thay đổi phần trăm**, gọi là **return**:

- **Simple return**: `r = (P_mới − P_cũ) / P_cũ`. Ví dụ giá đi từ 100 lên 110 thì r = +10%.
- **Log return**: `r = ln(P_mới / P_cũ)`. Từ 100 lên 110 thì r = ln(1.1) ≈ +9.53%.

Vì sao dùng **log return**? Vì nó **cộng được**: log return của 3 ngày liên tiếp = tổng log return từng ngày.
Code tính lợi nhuận cộng dồn rất gọn: `equity = exp(cumsum(log_return))`. Khi biến động nhỏ, hai loại return gần bằng nhau.

### 3.3 Vài đặc điểm của return crypto (xem notebook)

- Tỷ lệ ngày tăng khoảng **50–51%**, gần như tung đồng xu.
- **Đuôi dày (fat tails, kurtosis cao)**: những ngày tăng hoặc giảm 10–20% xuất hiện nhiều hơn hẳn so với phân phối chuẩn.
- **Return gần như không có tự tương quan**: hôm nay tăng không cho biết gì nhiều về ngày mai. Đó là lý do bài toán khó.
- **Volatility clustering**: biến động lớn thường đi liền nhau. Tuần đang biến động mạnh thì tuần sau cũng dễ biến động mạnh.
  Hướng giá khó đoán, nhưng *mức độ biến động* thì đoán được phần nào.

---

## 4. Giao dịch: long, flat, phí, Buy & Hold

### 4.1 Vị thế (position)

- **Long** (vị thế mua): đang giữ coin. Giá tăng thì lãi, giá giảm thì lỗ.
- **Flat** (đứng ngoài): đang giữ tiền mặt (USDT). Giá thay đổi thế nào cũng không lãi không lỗ.
- **Short** (bán khống): kiếm lời khi giá giảm. Project **không dùng** short cho đơn giản và ít rủi ro.

Chiến lược của project là **long/flat**: mỗi nến chỉ có hai trạng thái, 1 (giữ coin) hoặc 0 (giữ tiền).

### 4.2 Phí giao dịch (fee)

Mỗi lần mua hoặc bán, sàn thu phí, khoảng **0.1%** giá trị giao dịch (mức phí cơ bản trên Binance spot).
Nghe nhỏ, nhưng:

- Mua rồi bán (một vòng) mất 0.2%.
- Nếu giao dịch 1 000 lần, phí cộng lại có thể "ăn" gần như toàn bộ vốn: 0.999^1000 ≈ 0.37, tức mất 63% chỉ vì phí.

Đây là **phát hiện quan trọng nhất** của project (xem mục 13).

Ngoài phí còn có **slippage**: giá khớp thực tế thường tệ hơn một chút so với giá bạn thấy. Project chưa tính slippage.

### 4.3 Buy & Hold (B&H)

**Mua một lần và giữ mãi, không làm gì thêm.** Đây là **mốc so sánh (benchmark)** bắt buộc. Nếu chiến lược ML phức tạp
không tốt hơn B&H thì nó không có giá trị. Trong những năm crypto tăng mạnh, B&H rất khó bị đánh bại.

---

## 5. Chỉ báo kỹ thuật (feature)

**Chỉ báo kỹ thuật (technical indicator)** là các con số tính từ giá và khối lượng quá khứ, giúp mô tả trạng thái thị trường.
Trong ML, chúng được dùng làm **feature** (đặc trưng đầu vào). Project có 21 feature, chia nhóm như sau
(code ở `src/cq/features.py`):

| Nhóm | Feature | Ý nghĩa đơn giản |
|---|---|---|
| Return quá khứ | `ret_lag1..5` | Return của 1, 2, …, 5 nến gần nhất |
| Momentum | `mom_5/10/20` | Giá thay đổi bao nhiêu % trong 5/10/20 nến. Dương = đang có xu hướng tăng |
| Volatility | `vol_5/10/20` | Độ lệch chuẩn của return: đo mức độ "giật" của giá |
| RSI | `rsi_14` | Từ 0 tới 100. Trên 70 = "quá mua" (tăng nóng), dưới 30 = "quá bán" |
| MACD | `macd_hist` | Hiệu giữa EMA nhanh (12) và EMA chậm (26), so với đường signal (9). Đo đà tăng/giảm |
| Bollinger %B | `bb_pctb` | Giá đang ở đâu trong dải MA20 ± 2 độ lệch chuẩn. 0 = chạm dải dưới, 1 = chạm dải trên |
| ATR | `atr_pct` | Average True Range: biên độ dao động trung bình mỗi nến, tính theo % giá |
| Khoảng cách MA | `dist_ma20/50/200` | Giá cao hơn hay thấp hơn trung bình 20/50/200 nến bao nhiêu %. Trên MA200 thường được xem là xu hướng tăng dài hạn |
| Khối lượng | `volume_z` | Khối lượng hôm nay cao bất thường đến mức nào so với 20 nến trước (z-score) |
| Hình dạng nến | `hl_range`, `close_pos` | Nến dài hay ngắn; giá đóng cửa gần đỉnh hay gần đáy nến |

Một vài khái niệm phụ:

- **MA (Moving Average)**: trung bình giá N nến gần nhất. MA20 = trung bình 20 nến.
- **EMA (Exponential MA)**: trung bình nhưng ưu tiên các nến gần đây hơn.
- **z-score**: `(giá trị − trung bình) / độ lệch chuẩn`, cho biết giá trị đang "lệch" bao nhiêu so với bình thường.

**Quy tắc quan trọng nhất**: mọi feature tại nến t **chỉ được dùng dữ liệu tới nến t**, không được dùng dữ liệu tương lai.
Gọi là **causal**. Có test tự động kiểm tra quy tắc này (mục 9).

Các feature được LightGBM dùng nhiều nhất (BTC 1d): `vol_10`, `dist_ma50`, `atr_pct`, `vol_20`, `rsi_14`, `macd_hist`.
Nhóm volatility đứng đầu, khớp với nhận xét "biến động thì đoán được, hướng giá thì khó".

---

## 6. Machine learning cơ bản

### 6.1 Machine learning là gì?

Thay vì lập trình viên tự viết luật (kiểu "nếu RSI < 30 thì mua"), ta đưa cho máy **rất nhiều ví dụ có đáp án**,
và máy **tự tìm ra quy luật**. Cách này gọi là **học có giám sát (supervised learning)**.

Mỗi ví dụ gồm:
- **X (feature)**: thông tin đầu vào, ở đây là 21 chỉ báo tại nến t.
- **y (label / nhãn)**: đáp án đúng, ở đây là "giá sau 3 nến có cao hơn bây giờ không" (1 = có, 0 = không).

Bảng dữ liệu trông như sau (mỗi dòng là một nến):

| thời gian | rsi_14 | mom_10 | vol_20 | … | **y** |
|---|---:|---:|---:|---|---:|
| 2024-03-01 | 71.2 | 0.12 | 0.031 | … | 1 |
| 2024-03-02 | 74.5 | 0.15 | 0.033 | … | 0 |
| 2024-03-03 | 65.0 | 0.09 | 0.035 | … | 0 |

### 6.2 Phân loại (classification)

Vì y chỉ có hai giá trị 0/1, đây là bài toán **phân loại nhị phân**. Mô hình không chỉ trả lời 0/1 mà trả về **xác suất**
`p = P(giá tăng)`, ví dụ p = 0.58 nghĩa là "58% khả năng tăng". Từ p ta quyết định giao dịch (mục 10).

### 6.3 Train và test

- **Train (huấn luyện)**: cho mô hình xem dữ liệu kèm đáp án để nó học.
- **Test (kiểm tra)**: đưa dữ liệu **mô hình chưa từng thấy**, so dự đoán với đáp án thật.

Chỉ kết quả trên test mới có ý nghĩa. Một mô hình làm tốt trên train mà tệ trên test giống như học sinh học thuộc
đề cũ: gặp đề mới là sai.

### 6.4 Overfitting

**Overfitting (quá khớp)** là khi mô hình học cả **nhiễu ngẫu nhiên** trong dữ liệu train thay vì quy luật thật.
Dữ liệu tài chính rất nhiễu, nên overfitting là nguy cơ số 1. Cách chống:
- Dùng mô hình đơn giản, có **regularization** (phạt độ phức tạp).
- Đánh giá nghiêm túc trên dữ liệu chưa thấy (mục 9, 11).

### 6.5 Chuẩn hoá (scaling)

Các feature có thang đo khác nhau (RSI từ 0 tới 100, return khoảng ±0.05). Một số mô hình như Logistic Regression cần đưa về cùng thang
bằng **StandardScaler**: trừ trung bình, chia độ lệch chuẩn. Scaler cũng phải **chỉ fit trên train**.

---

## 7. Hai mô hình dùng trong project

### 7.1 Baseline: "luôn mua"

Một "mô hình" không học gì, lúc nào cũng nói "tăng". Giao dịch theo nó chính là Buy & Hold.
Mô hình ML thật **phải thắng baseline**, nếu không thì ML vô dụng.

### 7.2 Logistic Regression (LogReg)

Mô hình tuyến tính đơn giản nhất cho phân loại:

```
z = w1·rsi + w2·mom_10 + w3·vol_20 + ... + b         (tổng có trọng số)
p = 1 / (1 + e^(−z))                                  (hàm sigmoid, ép z về khoảng 0..1)
```

Máy học các trọng số `w` sao cho p khớp với y nhất. Ưu điểm: nhanh, khó overfit, dễ giải thích.
Nhược điểm: chỉ nắm được quan hệ "đường thẳng". Tham số `C=0.1` là regularization khá mạnh, buộc các trọng số nhỏ lại.

### 7.3 LightGBM (Gradient Boosting Decision Trees)

**Cây quyết định (decision tree)** là chuỗi câu hỏi có/không:

```
          vol_20 > 0.04 ?
          /            \
        có              không
        /                  \
  rsi_14 < 35 ?        dist_ma50 > 0 ?
   /      \              /       \
p=0.56   p=0.47      p=0.53    p=0.49
```

**Gradient boosting**: xây **hàng trăm cây nhỏ nối tiếp nhau**, mỗi cây mới sửa lỗi của các cây trước, rồi cộng kết quả lại.
**LightGBM** là một thư viện boosting nhanh của Microsoft. Nó nắm được quan hệ phi tuyến, ví dụ
"RSI thấp chỉ có ý nghĩa khi volatility cao".

Trong project, LightGBM được cố ý làm **"nhỏ"** để chống overfitting:
`num_leaves=8, max_depth=3` (cây nông), `min_child_samples=50` (mỗi lá cần ít nhất 50 mẫu), `learning_rate=0.03` (học chậm),
`subsample/colsample=0.8` (mỗi cây chỉ xem 80% dữ liệu và feature).

**Feature importance**: LightGBM cho biết feature nào được dùng nhiều để chia cây và giảm lỗi nhiều nhất (dashboard có biểu đồ này).

---

## 8. Đánh giá mô hình phân loại: accuracy, AUC, F1

| Chỉ số | Ý nghĩa | Mức "ngẫu nhiên" |
|---|---|---|
| **Accuracy** | % dự đoán đúng (lấy p > 0.5 là "tăng") | khoảng 50% (vì tỷ lệ tăng khoảng 50%) |
| **AUC** | Xác suất mô hình chấm điểm một nến tăng thật cao hơn một nến giảm thật. Không phụ thuộc ngưỡng | 0.5 |
| **F1** | Trung bình điều hoà của precision (khi báo "tăng" thì đúng bao nhiêu %) và recall (bắt được bao nhiêu % nến tăng) | phụ thuộc tỷ lệ lớp |
| `up_rate` | Tỷ lệ nến tăng thật trong tập test | – |

Kết quả project: accuracy khoảng **50–55%**, AUC khoảng **0.50–0.56**. Nghe thấp, nhưng với tài chính thì AUC 0.52–0.55 ổn định
đã là có tín hiệu. Nếu thấy accuracy 70–80% khi dự đoán giá, gần như chắc chắn là có lỗi leakage.

**Accuracy cao không có nghĩa là có lãi.** Một mô hình đoán đúng nhiều ngày tăng nhẹ nhưng sai đúng ngày sập 20%
vẫn có thể lỗ. Vì vậy phải backtest.

---

## 9. Cái bẫy lớn nhất: dữ liệu chuỗi thời gian và leakage

### 9.1 Leakage (rò rỉ dữ liệu) là gì?

**Leakage** = mô hình vô tình được "nhìn thấy tương lai" khi học hoặc khi dự đoán. Kết quả khi đó rất đẹp nhưng hoàn toàn giả.
Đây là lỗi phổ biến nhất trong các project dự đoán giá trên mạng. Một số dạng thường gặp:

| Lỗi | Ví dụ | Project xử lý thế nào |
|---|---|---|
| Feature dùng dữ liệu tương lai | Dùng `rolling(center=True)` hoặc chuẩn hoá bằng trung bình của **toàn bộ** dữ liệu | Mọi phép tính chỉ nhìn về quá khứ; có test kiểm tra (`tests/test_features.py`) |
| Chia train/test ngẫu nhiên | `train_test_split(shuffle=True)`: train có ngày 15/3 còn test có ngày 14/3 | Chia **theo thời gian**: train luôn nằm trước test |
| Fit scaler trên toàn bộ dữ liệu | `StandardScaler().fit(X_all)` | Scaler nằm trong pipeline, fit lại trên train của mỗi fold |
| Giao dịch bằng giá chưa biết | Dùng tín hiệu tính từ close nến t để mua tại open nến t | Tín hiệu tại **close** t, hưởng return từ close t tới close t+1 |
| Nhãn chồng lấn | Nhãn 3 nến tại t dùng giá t+1..t+3, có thể "rò" sang tập test | **Embargo**: bỏ 3 nến giữa train và test |
| Chọn tham số trên tập test | Thử 100 cấu hình rồi chọn cái tốt nhất trên test | Chọn trên **design**, kiểm tra một lần trên **holdout** (mục 11) |

**Test chống leakage** trong project: tạo dữ liệu giả, tính feature; sau đó **phá toàn bộ dữ liệu sau nến 300**
(nhân 3), tính lại feature. Nếu feature tại các nến 0..300 thay đổi, tức là feature đã dùng dữ liệu tương lai, test sẽ fail.

### 9.2 Walk-forward validation

Thay vì chia một lần 80/20, project dùng **walk-forward (expanding window)**, mô phỏng đúng cách dùng thật:

```
Fold 0: [====== train (2 năm) ======] gap [test 3 tháng]
Fold 1: [========= train ==============] gap [test 3 tháng]
Fold 2: [============ train =================] gap [test 3 tháng]
...                                                         ──► tới hiện tại
```

- Mỗi fold: **huấn luyện lại** mô hình trên toàn bộ quá khứ, rồi dự đoán 3 tháng tiếp theo.
- Ghép phần test của các fold lại thành một chuỗi dự đoán **out-of-sample** liên tục từ khoảng 2021 tới nay.
- Giống thực tế: mỗi quý bạn train lại mô hình với dữ liệu mới nhất rồi dùng cho quý sau.
- **gap / embargo**: bỏ vài nến giữa train và test để nhãn của train không "chạm" vào thời gian của test.

---

## 10. Backtest và các chỉ số hiệu quả

### 10.1 Backtest là gì?

**Backtest** = giả lập "nếu tôi đã giao dịch theo chiến lược này trong quá khứ thì kết quả sẽ ra sao", dùng dự đoán out-of-sample.

### 10.2 Từ xác suất tới vị thế: ngưỡng và hysteresis

Quy tắc đơn giản (v1): `p > 0.5 → long, ngược lại → flat`. Vấn đề là p thường dao động quanh 0.5
(0.51, 0.49, 0.52, …), nên chiến lược mua bán liên tục và mất rất nhiều phí.

**Hysteresis / vùng đệm** (v2, `band = 0.05`):
- Chỉ **mua** khi p > 0.55 (khá chắc sẽ tăng).
- Chỉ **bán** khi p < 0.45 (khá chắc sẽ giảm).
- Khi p ở giữa 0.45 và 0.55: **giữ nguyên** vị thế hiện tại.

Giống như điều hoà: bật khi nóng hơn 27°C, tắt khi mát hơn 25°C, chứ không bật tắt liên tục quanh 26°C.

```
p:      0.52  0.56  0.52  0.48  0.44  0.50
v1:       1     1     1     0     0     0     (đổi 1→0 tại 0.48)
v2:       0     1     1     1     0     0     (chỉ vào ở 0.56, chỉ ra ở 0.44)
```

### 10.3 Tính lợi nhuận

```
vị thế_t       = 1 hoặc 0, quyết định sau khi nến t đóng cửa
return_chiến_lược_t = vị thế_t × return(close_t → close_{t+1}) − phí nếu vị thế thay đổi
equity         = tích luỹ của (1 + return) theo thời gian, bắt đầu từ 1
```

**Equity curve** là đường giá trị tài khoản theo thời gian (vốn ban đầu = 1). Dashboard vẽ đường của mô hình cạnh đường của B&H.

### 10.4 Các chỉ số

| Chỉ số | Công thức / ý nghĩa | Đọc thế nào |
|---|---|---|
| **Total return** | equity cuối − 1 | +140% = vốn nhân 2.4 lần |
| **CAGR** | Lợi nhuận kép hằng năm: `equity^(1/số_năm) − 1` | So được giữa các giai đoạn dài ngắn khác nhau |
| **Sharpe ratio** | `trung bình return / độ lệch chuẩn return × √(số kỳ/năm)` | **Lợi nhuận trên mỗi đơn vị rủi ro.** < 0 tệ, 0.5 tạm, > 1 tốt, > 2 rất hiếm |
| **Max Drawdown (MDD)** | Mức sụt giảm lớn nhất từ đỉnh xuống đáy của equity | -77% = có lúc tài khoản mất 77% so với đỉnh trước đó |
| **Exposure** | % thời gian đang giữ coin | 60% = 40% thời gian đứng ngoài |
| **Win rate** | % nến có lãi trong những nến đang giữ coin | Không cần cao, miễn là lãi trung bình lớn hơn lỗ trung bình |
| **Số lệnh (n_trades)** | Số lần đổi vị thế | Càng nhiều thì càng mất phí |

Vì sao **Sharpe** và **Max Drawdown** quan trọng hơn total return? Hai chiến lược cùng lãi 100%, nhưng một cái
có lúc mất 80% còn cái kia chỉ mất 30%. Rõ ràng cái thứ hai tốt hơn: nó dễ "chịu đựng" về tâm lý, và có thể dùng đòn bẩy nhẹ để có lợi nhuận tương đương.

`√(số kỳ/năm)`: với 1d là √365, với 4h là √(365×6), dùng để quy Sharpe về theo năm.

---

## 11. Overfitting và cách đánh giá trung thực

### 11.1 Overfitting khi chọn cấu hình (data snooping)

Kể cả khi mô hình không overfit, **người làm** vẫn có thể overfit: thử 100 cấu hình (horizon, ngưỡng, mô hình, …)
rồi báo cáo cái tốt nhất. Do may mắn, kiểu gì cũng có một cấu hình trông rất đẹp, nhưng nó sẽ không lặp lại trong tương lai.

### 11.2 Design và holdout

Project chống lại điều này bằng cách chia thời gian làm hai phần:

```
2019 ───────────── 2024-12-31 │ 2025-01-01 ──────────── 2026-11
         DESIGN               │            HOLDOUT
  thử nghiệm, chọn cấu hình    │   không đụng tới khi chọn,
  (walk-forward bên trong)     │   chỉ xem MỘT lần ở cuối
```

Khi cải thiện từ v1 lên v2, khoảng 150 tổ hợp đã được thử, **nhưng chỉ đánh giá trên design**. Cấu hình được chọn
là cấu hình **đơn giản nhất trong nhóm tốt nhất**. Sau đó mới mở holdout ra xem. Nếu cải tiến chỉ là may mắn
thì nó sẽ không giữ được trên holdout. Ở đây nó giữ được (mục 14).

### 11.3 Ngưỡng `auto`

Tuỳ chọn `--threshold auto`: ở mỗi fold, lấy 25% cuối của **tập train** làm validation, thử các ngưỡng 0.45..0.60
và chọn ngưỡng có Sharpe tốt nhất **trên validation**, rồi dùng cho test. Tập test không bao giờ được dùng để chọn ngưỡng.

---

## 12. Đi qua từng file của project

```
crypto-quant/
├── config.yaml          ← mọi tham số: coin, khung, phí, horizon, band, holdout, kích thước walk-forward
├── src/cq/
│   ├── data.py          ← tải dữ liệu
│   ├── features.py      ← tính 21 chỉ báo
│   ├── labels.py        ← tạo nhãn y
│   ├── models.py        ← định nghĩa baseline, LogReg, LightGBM
│   ├── validation.py    ← walk-forward, chọn ngưỡng auto
│   ├── backtest.py      ← từ xác suất → vị thế → lợi nhuận
│   ├── metrics.py       ← tính accuracy, AUC, Sharpe, MDD, …
│   └── pipeline.py      ← nối tất cả lại, chạy bằng dòng lệnh
├── app.py               ← dashboard Streamlit
├── notebooks/eda.ipynb  ← phân tích dữ liệu và kết quả (dùng cho báo cáo)
└── tests/               ← kiểm tra tự động (pytest)
```

### `data.py`
- `fetch_ohlcv()`: gọi API sàn qua ccxt, mỗi lần lấy tối đa 1 000 nến, lặp (**phân trang**) cho tới hiện tại.
  `enableRateLimit=True` để không gọi quá nhanh làm sàn chặn.
- Bỏ **nến cuối nếu chưa đóng** (ví dụ đang 15h thì nến ngày hôm nay chưa kết thúc), vì giá của nó chưa phải giá cuối.
- `load_ohlcv()`: lưu dữ liệu ra file **parquet** trong `data/` (định dạng bảng nén, đọc nhanh), lần sau đọc từ file, không gọi API nữa.

### `features.py`
- `build_features(df)` trả về bảng 21 cột feature. Các dòng đầu sẽ có NaN (ví dụ MA200 cần 200 nến mới tính được), các dòng này bị bỏ ở bước sau.

### `labels.py`
- `forward_return(close, h)`: log return từ close_t tới close_{t+h}.
- `direction_label`: y = 1 nếu forward return > 0.
- `make_dataset`: ghép X, y và `fwd_ret` (return **1 nến** kế tiếp, dùng cho backtest), rồi bỏ dòng thiếu dữ liệu.
  Chú ý: **nhãn nhìn 3 nến** nhưng **backtest vẫn đánh giá lại mỗi nến**.

### `models.py`
- `make_model(name)` trả về mô hình theo tên: `baseline`, `logreg`, `lightgbm`.
- `feature_importance()` lấy mức quan trọng của feature (LightGBM: theo gain; LogReg: theo |hệ số|).

### `validation.py`
- `walk_forward_splits()` sinh các cặp (chỉ số train, chỉ số test) như sơ đồ ở mục 9.2.
- `run_walk_forward()`: với mỗi fold thì fit mô hình trên train, dự đoán xác suất trên test, ghi lại accuracy của fold và feature importance.
- `choose_threshold()`: ngưỡng `auto` (mục 11.3).

### `backtest.py`
- `positions(proba, threshold, band)`: biến xác suất thành chuỗi 0/1 có hysteresis.
- `backtest()`: tính số lệnh, trừ phí, tính return chiến lược, equity của chiến lược và của B&H.

### `metrics.py`
- `classification_metrics()`: accuracy, AUC, F1.
- `trading_metrics()`: total return, CAGR, Sharpe, MDD, exposure, win rate, số lệnh.

### `pipeline.py`
- `run_experiment()`: chạy toàn bộ chuỗi cho 1 coin × 1 khung × 1 mô hình.
- `run_all()`: chạy cho tất cả mô hình và thêm dòng B&H.
- `period_summary()`: tính metric riêng cho design và holdout.
- `main()`: giao diện dòng lệnh, ví dụ:

```bash
python -m cq.pipeline --symbol BTC/USDT --tf 1d                    # cấu hình mặc định (v2)
python -m cq.pipeline --symbol BTC/USDT --tf 4h --horizon 1 --band 0   # cấu hình v1
python -m cq.pipeline --symbol ETH/USDT --tf 1d --fee 0.0005 --threshold auto
```

### `tests/`
Chạy `pytest`. 13 test kiểm tra: feature không nhìn tương lai; nhãn đúng; luôn long với phí 0 thì bằng B&H; phí trừ đúng;
hysteresis đúng; walk-forward không chồng lấn và có embargo. **Nên nhắc tới các test này khi bảo vệ**:
chúng chứng minh kết quả không phải do lỗi code.

---

## 13. Từ v1 tới v2: project đã được cải thiện thế nào

### 13.1 Vấn đề của v1

v1: nhãn = hướng của **1 nến** kế tiếp, mua khi p > 0.5, bán khi p < 0.5.

- Ở khung 4h, mô hình có accuracy khoảng 54–55%, cao nhất trong các thử nghiệm. Nếu **không có phí**, BTC 4h lãi +1765%, Sharpe 1.4.
- Nhưng nó đổi vị thế khoảng **6 000 lần**. Với phí 0.1%, kết quả là **lỗ 95%**.

Kết luận: tín hiệu có thật, nhưng quá "vụn". Return của 1 nến rất nhiễu, nên p nhảy qua lại quanh 0.5 liên tục.

### 13.2 Hai thay đổi của v2

1. **Nhãn 3 nến (`horizon = 3`)**: dự đoán hướng giá sau 3 nến thay vì 1. Nhãn ổn định hơn, ít nhiễu hơn,
   nên dự đoán của các nến liền nhau cũng ổn định hơn. (Embargo cũng tăng lên 3 nến cho đúng.)
2. **Vùng đệm ±0.05 (`band = 0.05`)**: chỉ đổi vị thế khi mô hình đủ tự tin.

Kết quả với BTC 4h: số lệnh giảm từ 5 955 xuống 907; ở phí 0.1%, kết quả đi từ **-95% thành +305%**.

| Phí | 0% | 0.02% | 0.05% | 0.10% |
|---|---:|---:|---:|---:|
| v1 | +1765% | +467% | -5% | -95% |
| v2 | +903% | +736% | +537% | +305% |

v2 lãi ít hơn khi không có phí, nhưng **ổn định hơn rất nhiều khi phí tăng**. Đó mới là điều quan trọng ngoài đời thật.

### 13.3 Những gì đã thử mà không hiệu quả (nên kể trong báo cáo)

- **Làm mượt xác suất** (EMA của p): có giúp, nhưng vùng đệm đơn giản hơn mà hiệu quả tương đương.
- **Volatility targeting** (giảm vị thế khi biến động cao): giảm drawdown nhưng giảm Sharpe trung bình trên design, nên không dùng.
- **Mặc định giữ coin, chỉ bán khi chắc giảm** (ngưỡng bất đối xứng): kém hơn vùng đệm đối xứng.

Báo cáo những thứ "không được" cũng là một phần của phương pháp khoa học.

---

## 14. Đọc kết quả và kết luận

Trung bình trên 6 cặp (BTC/ETH/SOL × 1d/4h), phí 0.1%:

| | Design Sharpe | Design MDD | Holdout Sharpe | Holdout MDD | Thắng B&H (holdout) |
|---|---:|---:|---:|---:|---:|
| Buy & Hold | **0.97** | -72% | -0.11 | -66% | – |
| LightGBM v1 | 0.28 | -70% | -1.28 | -75% | 1/6 |
| **LightGBM v2** | 0.87 | -59% | **+0.10** | **-51%** | **4/6** |

Cách đọc bảng:

1. **v2 tốt hơn v1 rất nhiều, ở cả design lẫn holdout.** Holdout là dữ liệu không dùng khi chọn cấu hình,
   nên đây là bằng chứng cải tiến có thật chứ không phải do may mắn.
2. **Giai đoạn design (2021–2024) có nhiều đợt tăng mạnh**: B&H vẫn tốt nhất về Sharpe. Đứng ngoài thị trường thì bỏ lỡ một phần xu hướng tăng.
3. **Giai đoạn holdout (2025–2026) thị trường giảm**: B&H lỗ, còn v2 giảm lỗ hoặc có lãi (ETH +32% đến +49%), và drawdown nhỏ hơn rõ rệt.
4. Vì vậy, giá trị thực của mô hình là **quản lý rủi ro**: tự động đứng ngoài khi thị trường xấu. Nó không phải "máy in tiền".

**Hạn chế (nên tự nêu ra trước khi bị hỏi):**
- Holdout chỉ khoảng 22 tháng, và cả 3 coin tương quan mạnh với nhau, nên mẫu thực tế nhỏ hơn con số "6 cặp" gợi ra.
- Đã thử khoảng 150 tổ hợp trên design: vẫn còn rủi ro chọn nhầm do may mắn.
- Chưa tính slippage; chưa có short hay đòn bẩy; feature chỉ dựa trên giá và khối lượng.
- Kết quả quá khứ không đảm bảo tương lai.

---

## 15. Dùng dashboard

```bash
source .venv/bin/activate.fish      # hoặc .venv/bin/activate nếu dùng bash
streamlit run app.py                # mở http://localhost:8501
```

**Sidebar (bên trái):**
- *Cặp coin, Khung thời gian, Mô hình*: chọn thí nghiệm.
- *Tự chọn ngưỡng*: bật ngưỡng `auto` (mục 11.3).
- *Ngưỡng xác suất*: tâm của vùng đệm (mặc định 0.5).
- *Phí*: thử 0% để thấy phí ảnh hưởng thế nào.
- *Horizon*: 1 = như v1, 3 = v2.
- *Vùng đệm*: 0 = như v1, 0.05 = v2.

**Nội dung chính:**
- 6 ô chỉ số: lợi nhuận, CAGR, Sharpe, MDD của mô hình (dòng nhỏ bên dưới là của B&H), accuracy, số lệnh.
- Biểu đồ 3 tầng: giá (nến, thang log) với ▲ vào lệnh / ▼ thoát lệnh; equity so với B&H; drawdown.
- Bảng so sánh mọi mô hình với cùng cấu hình.
- Bảng **Design và holdout**.
- Accuracy từng fold (đường gạch = 50%) và feature importance.

**Gợi ý demo khi bảo vệ:**
1. Chọn BTC 4h, đặt horizon = 1, vùng đệm = 0, phí 0%: lãi rất lớn.
2. Kéo phí lên 0.1%: lỗ gần hết vốn. Đây là "vấn đề".
3. Đặt horizon = 3, vùng đệm = 0.05: số lệnh giảm mạnh, kết quả dương trở lại. Đây là "giải pháp".
4. Kéo xuống bảng Design và holdout: cải tiến vẫn giữ được trên dữ liệu chưa thấy.

---

## 16. Câu hỏi bảo vệ đồ án hay gặp

**Hỏi: Accuracy chỉ 52%, có khác gì tung đồng xu?**
Đáp: Tỷ lệ tăng khoảng 50%, nên 52–55% ổn định qua nhiều fold là có tín hiệu nhỏ. Quan trọng hơn, kết quả được đánh giá bằng
lợi nhuận sau phí, Sharpe và drawdown, so với Buy & Hold, chứ không chỉ bằng accuracy. Accuracy cao không đồng nghĩa có lãi.

**Hỏi: Sao không dùng LSTM hay Transformer?**
Đáp: Dữ liệu tài chính nhiễu và ít (khoảng 2 900 nến ngày), deep learning dễ overfit, và nhiều nghiên cứu cho thấy nó không
vượt trội hơn gradient boosting trên dữ liệu dạng bảng. Mô hình đơn giản dễ kiểm soát và giải thích hơn. Có thể làm phần mở rộng.

**Hỏi: Làm sao biết không có leakage?**
Đáp: (1) Feature causal, có test "phá tương lai". (2) Walk-forward, train luôn trước test, có embargo. (3) Scaler/model fit
lại trên train mỗi fold. (4) Tín hiệu tại close t, hưởng return t → t+1. (5) Chọn cấu hình chỉ trên design, holdout xem một lần.

**Hỏi: Tại sao chọn horizon 3 và band 0.05? Có phải chọn cho đẹp số không?**
Đáp: Hai thay đổi này có lý do rõ: giảm nhiễu của nhãn và giảm số lệnh, vì phân tích v1 cho thấy phí là vấn đề chính.
Cấu hình được chọn chỉ trên dữ liệu trước 2025, chọn cái đơn giản nhất trong nhóm tốt nhất, sau đó mới kiểm tra trên holdout 2025–2026.

**Hỏi: Mô hình không thắng Buy & Hold trong giai đoạn tăng, vậy có ích gì?**
Đáp: Nó cho Sharpe gần bằng B&H nhưng drawdown thấp hơn ở giai đoạn tăng, và thắng B&H khi thị trường giảm (holdout).
Giá trị nằm ở việc quản lý rủi ro. Kết luận trung thực này cũng là một kết quả nghiên cứu.

**Hỏi: Dữ liệu lấy từ đâu, có hợp pháp không?**
Đáp: Binance public market-data API qua ccxt, dữ liệu giá công khai, không có thông tin cá nhân, dùng cho mục đích học tập,
có ghi nguồn, không phân phối lại dữ liệu thô (repo không chứa dữ liệu).

**Hỏi: Nếu dùng thật thì cần thêm gì?**
Đáp: Tính slippage, kiểm tra trên nhiều coin/thị trường hơn, paper trading một thời gian, quản lý vốn (position sizing),
giám sát mô hình bị "lỗi thời" (concept drift), và giới hạn rủi ro.

**Hỏi: Embargo là gì, sao cần?**
Đáp: Nhãn tại t dùng giá tới t+3. Nếu nến cuối của train sát nến đầu của test thì nhãn của train "nhìn thấy" giá trong thời gian test.
Bỏ 3 nến ở giữa sẽ loại bỏ sự chồng lấn này.

---

## 17. Bảng thuật ngữ

| Thuật ngữ | Giải thích ngắn |
|---|---|
| API | Giao diện để chương trình lấy dữ liệu từ một dịch vụ (ở đây là sàn) |
| AUC | Khả năng xếp hạng nến tăng trên nến giảm; 0.5 = ngẫu nhiên |
| Backtest | Giả lập giao dịch trên dữ liệu quá khứ |
| Band / hysteresis | Vùng đệm quanh ngưỡng để tránh đổi vị thế liên tục |
| Baseline | Phương án đơn giản để so sánh (luôn mua) |
| Buy & Hold (B&H) | Mua một lần và giữ |
| CAGR | Tăng trưởng kép hằng năm |
| Causal | Chỉ dùng dữ liệu quá khứ và hiện tại |
| ccxt | Thư viện Python gọi API của nhiều sàn crypto |
| Design / holdout | Dữ liệu để chọn cấu hình / dữ liệu để kiểm tra lần cuối |
| Drawdown | Mức giảm từ đỉnh của tài khoản |
| Embargo / gap | Khoảng bỏ trống giữa train và test |
| Equity curve | Đường giá trị tài khoản theo thời gian |
| Exposure | % thời gian có vị thế |
| Feature | Đặc trưng đầu vào của mô hình |
| Fee | Phí giao dịch |
| Fold | Một lượt train/test trong walk-forward |
| Gradient boosting | Kết hợp nhiều cây quyết định, mỗi cây sửa lỗi cây trước |
| Horizon | Số nến tương lai dùng để tạo nhãn |
| Label (y) | Đáp án mô hình cần học |
| Leakage | Rò rỉ thông tin tương lai vào mô hình |
| LightGBM | Thư viện gradient boosting nhanh |
| Log return | ln(P_mới / P_cũ), cộng được theo thời gian |
| Logistic Regression | Mô hình phân loại tuyến tính + sigmoid |
| Long / flat / short | Giữ coin / giữ tiền / bán khống |
| MA / EMA | Trung bình trượt / trung bình trượt hàm mũ |
| Max Drawdown (MDD) | Mức giảm từ đỉnh lớn nhất |
| OHLCV | Open, High, Low, Close, Volume |
| Out-of-sample | Dữ liệu mô hình chưa thấy khi học |
| Overfitting | Học thuộc nhiễu, kém trên dữ liệu mới |
| Parquet | Định dạng file bảng nén |
| Regularization | Phạt độ phức tạp để chống overfit |
| Sharpe ratio | Lợi nhuận trên mỗi đơn vị rủi ro (theo năm) |
| Slippage | Chênh lệch giữa giá kỳ vọng và giá khớp thực tế |
| Stablecoin (USDT) | Coin neo giá khoảng 1 USD |
| StandardScaler | Chuẩn hoá về trung bình 0, độ lệch chuẩn 1 |
| Timeframe | Độ dài mỗi nến (1d, 4h) |
| Volatility | Mức độ biến động của giá (độ lệch chuẩn return) |
| Walk-forward | Train lại liên tục theo thời gian, test trên giai đoạn kế tiếp |
| Win rate | % kỳ có lãi khi đang giữ vị thế |

---

## 18. Đọc thêm

- Marcos López de Prado, *Advances in Financial Machine Learning* (2018): chương về leakage, purging/embargo, backtest overfitting.
- Ernest Chan, *Quantitative Trading* (2nd ed., 2021): nhập môn quant trading thực tế.
- Tài liệu scikit-learn: *Cross-validation of time series data* (`TimeSeriesSplit`).
- Tài liệu LightGBM: *Parameters Tuning*.
- Investopedia: tra nhanh các chỉ báo RSI, MACD, Bollinger Bands, ATR, Sharpe ratio.
