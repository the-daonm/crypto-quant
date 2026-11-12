"""Streamlit dashboard. Chạy: streamlit run app.py"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cq.data import load_ohlcv
from cq.models import MODEL_NAMES
from cq.pipeline import buy_and_hold_metrics, load_config, period_summary, run_experiment

st.set_page_config(page_title="Crypto Quant", layout="wide")
cfg = load_config()


@st.cache_data(show_spinner="Đang tải dữ liệu…")
def get_ohlcv(symbol: str, tf: str) -> pd.DataFrame:
    return load_ohlcv(symbol, tf, cfg["since"], cfg["exchange"])


@st.cache_data(show_spinner="Đang chạy walk-forward…")
def get_experiment(symbol: str, tf: str, model: str, threshold, fee: float, horizon: int, band: float):
    e = run_experiment(get_ohlcv(symbol, tf), tf, model, cfg, threshold, fee, horizon, band)
    return e.bt, e.wf.predictions, e.wf.folds, e.wf.importance, e.metrics


# ---------------- Sidebar ----------------
st.sidebar.header("Cấu hình")
symbol = st.sidebar.selectbox("Cặp coin", cfg["symbols"])
tf = st.sidebar.selectbox("Khung thời gian", cfg["timeframes"])
model = st.sidebar.selectbox("Mô hình", MODEL_NAMES, index=MODEL_NAMES.index("lightgbm"))
auto_thr = st.sidebar.checkbox("Tự chọn ngưỡng trên dữ liệu train", value=False)
thr = "auto" if auto_thr else st.sidebar.slider("Ngưỡng xác suất", 0.40, 0.65, float(cfg["threshold"]), 0.01)
fee_pct = st.sidebar.number_input("Phí mỗi lần đổi vị thế (%)", 0.0, 1.0, cfg["fee"] * 100, 0.01)
fee = fee_pct / 100
horizon = st.sidebar.select_slider("Horizon của nhãn (số nến)", [1, 2, 3, 6, 12], value=cfg["horizon"])
band = st.sidebar.slider("Vùng đệm quanh ngưỡng (hysteresis)", 0.0, 0.10, float(cfg["band"]), 0.01)
st.sidebar.caption("Dữ liệu: Binance public API qua ccxt. Chỉ dùng cho mục đích học tập.")

# ---------------- Run ----------------
bt, preds, folds, importance, metrics = get_experiment(symbol, tf, model, thr, fee, horizon, band)
ohlcv = get_ohlcv(symbol, tf).loc[bt.index[0]: bt.index[-1]]
bh = buy_and_hold_metrics(bt, tf)

st.title(f"{symbol} · {tf} · {model}")
st.caption(
    f"Out-of-sample (walk-forward) từ {bt.index[0].date()} tới {bt.index[-1].date()} · "
    f"{len(folds)} fold · Tín hiệu tại close nến t, giữ vị thế tới close nến t+1"
)

# ---------------- KPI ----------------
c = st.columns(6)
c[0].metric("Tổng lợi nhuận", f"{metrics['total_return']:.1%}", f"B&H {bh['total_return']:.1%}", delta_color="off")
c[1].metric("CAGR", f"{metrics['cagr']:.1%}", f"B&H {bh['cagr']:.1%}", delta_color="off")
c[2].metric("Sharpe", f"{metrics['sharpe']:.2f}", f"B&H {bh['sharpe']:.2f}", delta_color="off")
c[3].metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}", f"B&H {bh['max_drawdown']:.1%}", delta_color="off")
c[4].metric("Accuracy", f"{metrics['accuracy']:.1%}", f"tỷ lệ tăng {metrics['up_rate']:.1%}", delta_color="off")
c[5].metric("Số lệnh", f"{metrics['n_trades']:,}", f"thời gian giữ {metrics['exposure']:.0%}", delta_color="off")

# ---------------- Price + signals ----------------
pos = bt["position"]
entries = pos[(pos.diff() == 1) | ((pos == 1) & (pos.shift().isna()))].index
exits = pos[pos.diff() == -1].index

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.3, 0.2], vertical_spacing=0.03,
                    subplot_titles=("Giá và tín hiệu", "Equity (vốn ban đầu = 1)", "Drawdown"))
fig.add_trace(go.Candlestick(x=ohlcv.index, open=ohlcv["open"], high=ohlcv["high"], low=ohlcv["low"],
                             close=ohlcv["close"], name="Giá", showlegend=False), row=1, col=1)
if len(entries) <= 400:   # quá nhiều marker sẽ khó nhìn
    fig.add_trace(go.Scatter(x=entries, y=ohlcv.loc[entries, "close"], mode="markers", name="Vào lệnh",
                             marker=dict(symbol="triangle-up", size=8, color="#16a34a")), row=1, col=1)
    fig.add_trace(go.Scatter(x=exits, y=ohlcv.loc[exits, "close"], mode="markers", name="Thoát lệnh",
                             marker=dict(symbol="triangle-down", size=8, color="#dc2626")), row=1, col=1)
fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name=model, line=dict(color="#2563eb")), row=2, col=1)
fig.add_trace(go.Scatter(x=bt.index, y=bt["bh_equity"], name="Buy & Hold", line=dict(color="#9ca3af")), row=2, col=1)
for col, name, color in (("equity", model, "#2563eb"), ("bh_equity", "B&H", "#9ca3af")):
    dd = bt[col] / bt[col].cummax() - 1
    fig.add_trace(go.Scatter(x=bt.index, y=dd, name=f"DD {name}", line=dict(color=color), showlegend=False),
                  row=3, col=1)
fig.update_layout(height=850, xaxis_rangeslider_visible=False, margin=dict(t=40, b=10),
                  legend=dict(orientation="h", y=1.04))
fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(tickformat=".0%", row=3, col=1)
st.plotly_chart(fig, width="stretch")

# ---------------- Comparison table ----------------
st.subheader("So sánh các mô hình (cùng ngưỡng và phí)")
all_runs = {m: get_experiment(symbol, tf, m, thr, fee, horizon, band) for m in MODEL_NAMES}
rows = [bh] + [r[4] for r in all_runs.values()]
table = pd.DataFrame(rows).set_index("model")
st.dataframe(
    table.style.format({
        "total_return": "{:.1%}", "cagr": "{:.1%}", "sharpe": "{:.2f}", "max_drawdown": "{:.1%}",
        "accuracy": "{:.1%}", "auc": "{:.3f}", "f1": "{:.3f}", "up_rate": "{:.1%}",
        "exposure": "{:.0%}", "win_rate": "{:.1%}", "n_trades": "{:,.0f}",
    }, na_rep="–"),
    width="stretch",
)

# ---------------- Design vs holdout ----------------
hs = cfg.get("holdout_start")
if hs:
    bts = {m: r[0] for m, r in all_runs.items()}
    cols = ["total_return", "cagr", "sharpe", "max_drawdown", "n_trades"]
    fmt = {
        "total_return": "{:.1%}", "cagr": "{:.1%}", "sharpe": "{:.2f}",
        "max_drawdown": "{:.1%}", "n_trades": "{:,.0f}",
    }
    st.subheader("Design và holdout")
    st.caption(f"Cấu hình mặc định được chọn chỉ dựa trên dữ liệu trước {hs}. "
               "Holdout là dữ liệu chưa từng dùng để chọn cấu hình, nên đây là phép thử trung thực nhất.")
    a, b = st.columns(2)
    a.markdown(f"**Design** (< {hs})")
    a.dataframe(period_summary(bts, tf, end=hs)[cols].style.format(fmt, na_rep="–"), width="stretch")
    b.markdown(f"**Holdout** (≥ {hs})")
    b.dataframe(period_summary(bts, tf, start=hs)[cols].style.format(fmt, na_rep="–"), width="stretch")

# ---------------- Folds + importance ----------------
left, right = st.columns(2)
with left:
    st.subheader("Accuracy theo từng fold")
    ff = go.Figure(go.Bar(x=folds["test_start"], y=folds["accuracy"], marker_color="#2563eb"))
    ff.add_hline(y=0.5, line_dash="dash", line_color="#9ca3af")
    ff.update_layout(height=350, yaxis_tickformat=".0%", margin=dict(t=10, b=10))
    st.plotly_chart(ff, width="stretch")
with right:
    st.subheader("Mức quan trọng của feature")
    if importance is None:
        st.info("Baseline không có feature importance.")
    else:
        top = importance.head(15)[::-1]
        fi = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color="#2563eb"))
        fi.update_layout(height=350, xaxis_tickformat=".0%", margin=dict(t=10, b=10))
        st.plotly_chart(fi, width="stretch")

with st.expander("Chi tiết các fold walk-forward"):
    st.dataframe(folds, width="stretch")
