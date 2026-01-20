# app.py
"""
Time-Series Forecasting and Model Comparison (Streamlit)

Methodology:
- SARIMA / SARIMAX (primary seasonal model)
- ARIMA / ARIMAX (non-seasonal benchmark)
- Seasonal Naïve baseline
- Prophet (optional benchmark, if installed)

Input files (dataset folder):
- transactions.csv (required): date, store_nbr, transactions
- oil.csv (optional): date, dcoilwtico (or one numeric column)
- holidays_events.csv (optional): date (used to create holiday indicators)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
plt.close("all")  # helps when Streamlit reruns

# Optional Prophet
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# ----------------------------
# Metrics + transforms
# ----------------------------
def rmse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat, eps=1e-9):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.maximum(np.abs(y), eps)
    return float(np.mean(np.abs((y - yhat) / denom)) * 100.0)


def safe_log1p(x):
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))


def safe_expm1(x):
    return np.expm1(np.asarray(x, dtype=float))


# ----------------------------
# Data loader
# ----------------------------
@st.cache_data(show_spinner=False)
def load_favorita(folder: str) -> pd.DataFrame:
    folder = Path(folder)

    transactions_path = folder / "transactions.csv"
    oil_path = folder / "oil.csv"
    holidays_path = folder / "holidays_events.csv"

    if not transactions_path.exists():
        raise FileNotFoundError(f"Missing required file: {transactions_path}")

    trans = pd.read_csv(transactions_path, parse_dates=["date"])

    needed = {"date", "store_nbr", "transactions"}
    if not needed.issubset(set(trans.columns)):
        raise ValueError(
            f"transactions.csv must contain columns {needed}. Found: {list(trans.columns)}"
        )

    # Oil optional
    oil = None
    if oil_path.exists():
        oil_tmp = pd.read_csv(oil_path, parse_dates=["date"])
        if "dcoilwtico" not in oil_tmp.columns:
            candidates = [c for c in oil_tmp.columns if c != "date"]
            if len(candidates) == 1:
                oil_tmp = oil_tmp.rename(columns={candidates[0]: "dcoilwtico"})
            else:
                oil_tmp = None
        oil = oil_tmp

    # Holidays optional (aggregate to date)
    hol_agg = None
    if holidays_path.exists():
        hol = pd.read_csv(holidays_path, parse_dates=["date"]).copy()
        hol["holiday_any"] = 1
        hol_agg = (
            hol.groupby("date", as_index=False)
            .agg(
                holiday_any=("holiday_any", "max"),
                holiday_event_count=("holiday_any", "sum"),
            )
        )

    df = trans.copy()

    if oil is not None:
        df = df.merge(oil[["date", "dcoilwtico"]], on="date", how="left").sort_values("date")
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    else:
        df["dcoilwtico"] = 0.0

    if hol_agg is not None:
        df = df.merge(hol_agg, on="date", how="left")
        df["holiday_any"] = df["holiday_any"].fillna(0).astype(int)
        df["holiday_event_count"] = df["holiday_event_count"].fillna(0).astype(int)
    else:
        df["holiday_any"] = 0
        df["holiday_event_count"] = 0

    return df


# ----------------------------
# Build daily enterprise series (+ features)
# ----------------------------
def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            transactions=("transactions", "sum"),
            dcoilwtico=("dcoilwtico", "mean"),
            holiday_any=("holiday_any", "max"),
            holiday_event_count=("holiday_event_count", "sum"),
        )
        .sort_values("date")
    )

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["log_transactions"] = safe_log1p(daily["transactions"].values)
    return daily


def time_split(daily: pd.DataFrame, train_ratio: float = 0.8):
    n = len(daily)
    split_idx = int(n * train_ratio)
    return daily.iloc[:split_idx].copy(), daily.iloc[split_idx:].copy(), split_idx


def seasonal_naive_forecast(daily: pd.DataFrame, split_idx: int, season: int = 7) -> np.ndarray:
    full = daily.reset_index(drop=True).copy()
    full["naive"] = full["transactions"].shift(season)
    return full.iloc[split_idx:]["naive"].bfill().to_numpy()


def slice_last_weeks(daily: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Return last N weeks from daily (by date)."""
    last_date = pd.to_datetime(daily["date"].max())
    start_date = last_date - pd.Timedelta(weeks=int(weeks))
    return daily[daily["date"] >= start_date].copy()


# ----------------------------
# Plots
# ----------------------------
def fig_timeseries(daily, title="Daily Transactions"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily["date"], daily["transactions"])
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Transactions")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_distribution(daily, title="Distribution of Daily Transactions"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(daily["transactions"].values, bins=30)
    ax.set_title(title)
    ax.set_xlabel("Transactions")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def fig_decomposition(daily, period=7, use_log=True, title="Seasonal Decomposition"):
    series = daily.set_index("date")["log_transactions" if use_log else "transactions"].asfreq("D")
    series = series.interpolate().ffill().bfill()
    decomp = seasonal_decompose(series, model="additive", period=period)
    fig = decomp.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def fig_acf_pacf(daily, lags=30, use_log=True, title_suffix=""):
    series = daily["log_transactions" if use_log else "transactions"].values

    fig1, ax1 = plt.subplots(figsize=(9, 3.2))
    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title(f"Autocorrelation Function (ACF){title_suffix}")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    plot_pacf(series, lags=lags, ax=ax2, method="ywm")
    ax2.set_title(f"Partial Autocorrelation Function (PACF){title_suffix}")
    fig2.tight_layout()

    return fig1, fig2


def fig_weekly_monthly_resample(daily, how="sum", title_prefix=""):
    s = daily.set_index("date")["transactions"].asfreq("D").interpolate().ffill().bfill()

    if how == "mean":
        weekly = s.resample("W-SUN").mean()
        monthly = s.resample("M").mean()
        title_w = f"{title_prefix}Weekly Average Transactions"
        title_m = f"{title_prefix}Monthly Average Transactions"
        ylabel = "Average transactions"
    else:
        weekly = s.resample("W-SUN").sum()
        monthly = s.resample("M").sum()
        title_w = f"{title_prefix}Weekly Total Transactions"
        title_m = f"{title_prefix}Monthly Total Transactions"
        ylabel = "Total transactions"

    figw, axw = plt.subplots(figsize=(11, 3.6))
    axw.plot(weekly.index, weekly.values)
    axw.set_title(title_w)
    axw.set_xlabel("Week")
    axw.set_ylabel(ylabel)
    axw.grid(True, alpha=0.2)
    figw.tight_layout()

    figm, axm = plt.subplots(figsize=(11, 3.6))
    axm.plot(monthly.index, monthly.values)
    axm.set_title(title_m)
    axm.set_xlabel("Month")
    axm.set_ylabel(ylabel)
    axm.grid(True, alpha=0.2)
    figm.tight_layout()

    return figw, figm


def fig_seasonal_profile_weekly_monthly(daily, title_prefix=""):
    tmp = daily.copy()
    tmp["dow"] = tmp["date"].dt.day_name()
    tmp["month"] = tmp["date"].dt.month_name()

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    dow_avg = tmp.groupby("dow")["transactions"].mean().reindex(dow_order)
    mon_avg = tmp.groupby("month")["transactions"].mean().reindex(month_order)

    fig1, ax1 = plt.subplots(figsize=(9, 3.6))
    ax1.plot(dow_avg.index, dow_avg.values, marker="o")
    ax1.set_title(f"{title_prefix}Weekday Profile (Average Transactions)")
    ax1.set_xlabel("Weekday")
    ax1.set_ylabel("Average transactions")
    ax1.grid(True, alpha=0.2)
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(11, 3.6))
    ax2.plot(mon_avg.index, mon_avg.values, marker="o")
    ax2.set_title(f"{title_prefix}Monthly Profile (Average Transactions)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Average transactions")
    ax2.grid(True, alpha=0.2)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
    fig2.tight_layout()

    return fig1, fig2


def summarize_seasonality(daily):
    tmp = daily.copy()
    tmp["dow"] = tmp["date"].dt.day_name()
    tmp["month_num"] = tmp["date"].dt.month
    tmp["month_name"] = tmp["date"].dt.month_name()

    dow_means = tmp.groupby("dow")["transactions"].mean()
    best_dow = dow_means.idxmax()
    worst_dow = dow_means.idxmin()

    mon_means = tmp.groupby(["month_num", "month_name"])["transactions"].mean().sort_index()
    best_mon = mon_means.idxmax()[1]
    worst_mon = mon_means.idxmin()[1]

    return best_dow, worst_dow, best_mon, worst_mon


# ----------------------------
# Models: ARIMA / SARIMAX (log-space)
# ----------------------------
def fit_arimax_log(y_train_log, X_train, X_test, order=(1, 1, 1)):
    model = ARIMA(y_train_log, order=order, exog=X_train)
    res = model.fit()
    fc = res.get_forecast(steps=len(X_test), exog=X_test)
    pred_log = np.asarray(fc.predicted_mean)
    return res, pred_log


def fit_sarimax_log(y_train_log, X_train, X_test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    model = SARIMAX(
        y_train_log,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=200)
    fc = res.get_forecast(steps=len(X_test), exog=X_test)
    pred_log = np.asarray(fc.predicted_mean)
    ci_log = np.asarray(fc.conf_int(alpha=0.05))  # 95% PI (log space)
    return res, pred_log, ci_log


# ----------------------------
# Model: Prophet (log-space)
# ----------------------------
def _make_prophet_df(dates, y):
    return pd.DataFrame({"ds": pd.to_datetime(dates), "y": np.asarray(y, dtype=float)})


def fit_prophet_log(train_df: pd.DataFrame, test_df: pd.DataFrame, exog_cols: list, interval_width: float = 0.80):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed. Run: pip install prophet")

    y_train_log = train_df["log_transactions"].astype(float).values
    dfp_train = _make_prophet_df(train_df["date"], y_train_log)

    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=float(interval_width),
    )

    for c in exog_cols:
        m.add_regressor(c)
        dfp_train[c] = train_df[c].astype(float).values

    m.fit(dfp_train)

    future = pd.DataFrame({"ds": pd.to_datetime(test_df["date"])})
    for c in exog_cols:
        future[c] = test_df[c].astype(float).values

    fc = m.predict(future)
    pred_log = fc["yhat"].to_numpy()
    lower_log = fc["yhat_lower"].to_numpy()
    upper_log = fc["yhat_upper"].to_numpy()
    return m, pred_log, lower_log, upper_log


# ----------------------------
# Export helper
# ----------------------------
def save_fig(fig, out_dir: Path, filename: str, dpi: int = 300) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="IT Prototype: Big data Time-Series Forecasting Analysis and Visualization", layout="wide")
st.title("IT Prototype: Big data Time-Series Forecasting Analysis and Visualization")
st.caption("Primary model: SARIMA/SARIMAX. Benchmarks: Seasonal Naïve, ARIMA/ARIMAX. Prophet optional.")

with st.sidebar:
    st.header("Dataset")
    data_folder = st.text_input("Dataset folder path", value=r"D:\Thesis prototype\Dataset")

    st.header("Train/Test")
    train_ratio = st.slider("Train ratio (time-based)", 0.60, 0.90, 0.80, 0.05)
    st.caption(f"Train: {int(train_ratio*100)}% | Test: {int((1-train_ratio)*100)}%")

    st.header("Seasonality Settings")
    season_period = st.selectbox("Season length (days)", [7, 14], index=0)
    lags = st.slider("ACF/PACF lags", 10, 60, 30, 5)

    st.header("Resampling")
    resample_mode = st.selectbox("Aggregation", ["sum", "mean"], index=0)
    show_seasonal_profiles = st.checkbox("Seasonal profiles (weekday/month)", value=True)
    show_seasonality_text = st.checkbox("Seasonality summary", value=True)

    st.header("Zoom Window")
    show_zoom = st.checkbox("Enable zoom view", value=True)
    zoom_weeks = st.slider("Weeks", 8, 12, 12)
    show_acf_full = st.checkbox("ACF/PACF (full series)", value=True)
    show_acf_zoom = st.checkbox("ACF/PACF (zoom series)", value=True)

    st.header("Model Parameters")
    arima_order = st.selectbox(
        "ARIMA/ARIMAX (p,d,q)",
        [(1, 1, 1), (0, 1, 1), (1, 0, 1), (2, 1, 2)],
        index=0,
    )
    sarima_order = st.selectbox("SARIMA/SARIMAX (p,d,q)", [(1, 1, 1), (0, 1, 1), (1, 0, 1)], index=0)
    sarima_seasonal = st.selectbox("Seasonal (P,D,Q,s)", [(1, 1, 1, 7), (1, 0, 1, 7), (0, 1, 1, 7)], index=0)

    st.header("Exogenous Variables")
    use_dcoil = st.checkbox("Oil price (dcoilwtico)", value=True)
    use_holiday_any = st.checkbox("Holiday indicator", value=True)
    use_holiday_count = st.checkbox("Holiday event count", value=True)
    use_weekend = st.checkbox("Weekend indicator", value=True)

    st.header("Prophet (Optional)")
    enable_prophet = st.checkbox("Enable Prophet", value=False)
    prophet_interval = st.select_slider("Interval width", options=[0.80, 0.90, 0.95], value=0.80)
    if enable_prophet and not PROPHET_AVAILABLE:
        st.warning("Prophet not installed. Install with: pip install prophet")

    st.header("Export")
    out_dir_name = st.text_input("Save folder name", value="thesis_outputs")
    export_on = st.checkbox("Enable export (PNG/CSV)", value=True)

# Load data
try:
    raw = load_favorita(data_folder)
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.info("Check the path and ensure transactions.csv exists in the selected folder.")
    st.stop()

daily = build_daily(raw)

# Zoom slice (not used for training)
daily_zoom = slice_last_weeks(daily, zoom_weeks) if show_zoom else None

# Train/test on full data
train, test, split_idx = time_split(daily, train_ratio=train_ratio)

# Build exog columns based on toggles
exog_cols = []
if use_dcoil:
    exog_cols.append("dcoilwtico")
if use_holiday_any:
    exog_cols.append("holiday_any")
if use_holiday_count:
    exog_cols.append("holiday_event_count")
if use_weekend:
    exog_cols.append("is_weekend")

if len(exog_cols) == 0:
    X_train = np.zeros((len(train), 1), dtype=float)
    X_test = np.zeros((len(test), 1), dtype=float)
else:
    X_train = train[exog_cols].astype(float).values
    X_test = test[exog_cols].astype(float).values

out_dir = Path(data_folder) / out_dir_name

# ----------------------------
# 1. Dataset Overview
# ----------------------------
st.header("1. Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Days", f"{len(daily):,}")
c2.metric("Start date", str(daily["date"].min().date()))
c3.metric("End date", str(daily["date"].max().date()))
c4.metric("Mean", f"{daily['transactions'].mean():.2f}")

st.subheader("1.1 Data Preview")
st.dataframe(daily.head(20), width="stretch")
st.write("Exogenous variables:", exog_cols if exog_cols else ["(none)"])

# ----------------------------
# 2. Series Overview (Full Sample)
# ----------------------------
st.header("2. Series Overview (Full Sample)")

colA, colB = st.columns(2)
with colA:
    fig_ts = fig_timeseries(daily, title="Daily Transactions (Full Sample)")
    st.pyplot(fig_ts, width="stretch")
    st.caption("Full sample time series.")
with colB:
    fig_hist = fig_distribution(daily, title="Distribution of Daily Transactions")
    st.pyplot(fig_hist, width="stretch")
    st.caption("Empirical distribution of daily totals.")

st.subheader("2.1 Resampled Views (Weekly/Monthly)")
fig_w, fig_m = fig_weekly_monthly_resample(daily, how=resample_mode, title_prefix="")
st.pyplot(fig_w, width="stretch")
st.caption("Weekly aggregation.")
st.pyplot(fig_m, width="stretch")
st.caption("Monthly aggregation.")

if show_seasonal_profiles:
    st.subheader("2.2 Seasonal Profiles (Weekday/Month)")
    fig_dow, fig_month = fig_seasonal_profile_weekly_monthly(daily, title_prefix="")
    st.pyplot(fig_dow, width="stretch")
    st.caption("Average by weekday.")
    st.pyplot(fig_month, width="stretch")
    st.caption("Average by month.")

if show_seasonality_text:
    best_dow, worst_dow, best_mon, worst_mon = summarize_seasonality(daily)
    st.info(
        f"Weekday: highest = **{best_dow}**, lowest = **{worst_dow}** | "
        f"Month: highest = **{best_mon}**, lowest = **{worst_mon}**."
    )

# ----------------------------
# 2B. Zoom View (Last N Weeks)
# ----------------------------
if daily_zoom is not None and len(daily_zoom) > 0:
    st.header(f"2B. Zoom View (Last {zoom_weeks} Weeks)")
    z1, z2 = st.columns(2)
    with z1:
        st.pyplot(fig_timeseries(daily_zoom, title=f"Daily Transactions (Last {zoom_weeks} weeks)"), width="stretch")
        st.caption("Short window view for local pattern inspection.")
    with z2:
        st.pyplot(fig_distribution(daily_zoom, title="Distribution (Zoom View)"), width="stretch")
        st.caption("Empirical distribution (zoom view).")

    st.subheader("2B.1 Weekly Totals (Zoom View)")
    fig_wz, _ = fig_weekly_monthly_resample(daily_zoom, how="sum", title_prefix="")
    st.pyplot(fig_wz, width="stretch")
    st.caption("Weekly totals in the zoom window.")

    if show_seasonal_profiles:
        st.subheader("2B.2 Weekday Profile (Zoom View)")
        fig_dow_z, _ = fig_seasonal_profile_weekly_monthly(daily_zoom, title_prefix="")
        st.pyplot(fig_dow_z, width="stretch")
        st.caption("Average by weekday (zoom view).")
        # ----------------------------
# 2C. Seasonal Decomposition Zoom (Last N Weeks)
# ----------------------------
if daily_zoom is not None and len(daily_zoom) > 0:
    st.header(f"2C. Seasonal Decomposition Zoom (Last {zoom_weeks} Weeks)")

    # Use log-transformed transactions for decomposition
    series_zoom = daily_zoom.set_index("date")["log_transactions"].asfreq("D").interpolate().ffill().bfill()

    decomp_zoom = seasonal_decompose(series_zoom, model="additive", period=season_period)

    fig_dec_zoom = decomp_zoom.plot()
    fig_dec_zoom.set_size_inches(11, 7)
    fig_dec_zoom.suptitle(f"Seasonal Decomposition (Log scale, Last {zoom_weeks} Weeks)", y=1.02)
    fig_dec_zoom.tight_layout()

    st.pyplot(fig_dec_zoom, width="stretch")

    # Optional: export zoom decomposition
    if export_on:
        save_fig(fig_dec_zoom, out_dir, f"Decomposition_Zoom_{zoom_weeks}w.png")


# ----------------------------
# 3. Seasonality Diagnostics
# ----------------------------
st.header("3. Seasonality Diagnostics")

st.subheader("3.1 Seasonal Decomposition")
fig_dec = fig_decomposition(daily, period=season_period, use_log=True, title="Seasonal Decomposition (Log scale)")
st.pyplot(fig_dec, width="stretch")
st.caption("Seasonal component indicates repeating structure; residual captures irregular variation.")

st.subheader("3.2 Autocorrelation Diagnostics (ACF/PACF)")
if show_acf_full:
    st.markdown("**Full sample**")
    fig_acf_full, fig_pacf_full = fig_acf_pacf(daily, lags=lags, use_log=True, title_suffix="")
    st.pyplot(fig_acf_full, width="stretch")
    st.caption("Weekly seasonality is indicated by spikes at lags 7, 14, 21, ...")
    st.pyplot(fig_pacf_full, width="stretch")

if show_acf_zoom and (daily_zoom is not None) and len(daily_zoom) > 30:
    st.markdown(f"**Zoom view (last {zoom_weeks} weeks)**")
    fig_acf_z, fig_pacf_z = fig_acf_pacf(daily_zoom, lags=min(lags, 30), use_log=False, title_suffix="")
    st.pyplot(fig_acf_z, width="stretch")
    st.caption("Weekly seasonality is indicated by spikes at lags 7, 14, 21, ...")
    st.pyplot(fig_pacf_z, width="stretch")

# ----------------------------
# 4. Train/Test Split
# ----------------------------
st.header("4. Train/Test Split (Time-Based)")
st.write(f"Split index: **{split_idx}** | Train size: **{len(train)}** | Test size: **{len(test)}**")
st.dataframe(test.head(10), width="stretch")

# ----------------------------
# 5. Forecasting Models
# ----------------------------
st.header("5. Forecasting Models")
y_test = test["transactions"].values.astype(float)
y_train_log = train["log_transactions"].values.astype(float)

st.subheader("5.1 Seasonal Naïve Baseline")
naive_pred = seasonal_naive_forecast(daily, split_idx, season=season_period)

fig_naive, ax = plt.subplots(figsize=(11, 4))
ax.plot(test["date"], y_test, label="Observed")
ax.plot(test["date"], naive_pred, label=f"Seasonal naive (lag {season_period})")
ax.set_title("Seasonal Naïve: Observed vs Forecast (Test Window)")
ax.set_xlabel("Date")
ax.set_ylabel("Transactions")
ax.legend()
fig_naive.tight_layout()
st.pyplot(fig_naive, width="stretch")
st.caption("Baseline forecast for comparison.")

st.subheader("5.2 ARIMA / ARIMAX")
with st.spinner("Fitting ARIMA / ARIMAX..."):
    arima_res, arima_pred_log = fit_arimax_log(y_train_log, X_train, X_test, order=arima_order)
arima_pred = safe_expm1(arima_pred_log)

fig_arima, ax = plt.subplots(figsize=(11, 4))
ax.plot(test["date"], y_test, label="Observed")
ax.plot(test["date"], arima_pred, label="ARIMA/ARIMAX")
ax.set_title("ARIMA/ARIMAX: Observed vs Forecast (Test Window)")
ax.set_xlabel("Date")
ax.set_ylabel("Transactions")
ax.legend()
fig_arima.tight_layout()
st.pyplot(fig_arima, width="stretch")

with st.expander("ARIMA/ARIMAX Model Summary"):
    st.text(str(arima_res.summary()))

st.subheader("5.3 SARIMA / SARIMAX (Primary)")
with st.spinner("Fitting SARIMA / SARIMAX..."):
    sarimax_res, sarimax_pred_log, sarimax_ci_log = fit_sarimax_log(
        y_train_log,
        X_train,
        X_test,
        order=sarima_order,
        seasonal_order=sarima_seasonal,
    )

sarimax_pred = safe_expm1(sarimax_pred_log)
sarimax_ci = safe_expm1(np.asarray(sarimax_ci_log))
lower, upper = sarimax_ci[:, 0], sarimax_ci[:, 1]

fig_sarima, ax = plt.subplots(figsize=(11, 4))
ax.plot(test["date"], y_test, label="Observed")
ax.plot(test["date"], sarimax_pred, label="SARIMA/SARIMAX")
ax.fill_between(test["date"], lower, upper, alpha=0.2, label="95% prediction interval")
ax.set_title("SARIMA/SARIMAX: Forecast with 95% Prediction Interval (Test Window)")
ax.set_xlabel("Date")
ax.set_ylabel("Transactions")
ax.legend()
fig_sarima.tight_layout()
st.pyplot(fig_sarima, width="stretch")

with st.expander("SARIMA/SARIMAX Model Summary"):
    st.text(str(sarimax_res.summary()))

# Prophet block (optional benchmark)
prophet_pred = None
prophet_lo = None
prophet_hi = None
prophet_model = None
fig_prophet = None
fig_prophet_components = None

if enable_prophet:
    st.subheader("5.4 Prophet (Optional Benchmark)")
    if not PROPHET_AVAILABLE:
        st.error("Prophet is not installed. Install with: pip install prophet (or disable Prophet).")
    else:
        with st.spinner("Fitting Prophet..."):
            prophet_model, prophet_pred_log, prophet_lo_log, prophet_hi_log = fit_prophet_log(
                train_df=train,
                test_df=test,
                exog_cols=exog_cols,
                interval_width=float(prophet_interval),
            )

        prophet_pred = safe_expm1(prophet_pred_log)
        prophet_lo = safe_expm1(prophet_lo_log)
        prophet_hi = safe_expm1(prophet_hi_log)

        fig_prophet, ax = plt.subplots(figsize=(11, 4))
        ax.plot(test["date"], y_test, label="Observed")
        ax.plot(test["date"], prophet_pred, label="Prophet")
        ax.fill_between(
            test["date"],
            prophet_lo,
            prophet_hi,
            alpha=0.2,
            label=f"{int(float(prophet_interval)*100)}% interval",
        )
        ax.set_title("Prophet: Observed vs Forecast (Test Window)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Transactions")
        ax.legend()
        fig_prophet.tight_layout()
        st.pyplot(fig_prophet, width="stretch")

        with st.expander("Prophet components (trend / weekly / yearly)"):
            full_future = pd.DataFrame({"ds": pd.to_datetime(daily["date"])})
            for c in exog_cols:
                full_future[c] = daily[c].astype(float).values
            full_fc = prophet_model.predict(full_future)
            fig_prophet_components = prophet_model.plot_components(full_fc)
            st.pyplot(fig_prophet_components, width="stretch")
            # ----------------------------
# 5B. SARIMA/SARIMAX Zoom Forecast (Last N Weeks)
# ----------------------------
if daily_zoom is not None and len(daily_zoom) > 0:
    st.subheader(f"5B. SARIMA/SARIMAX Zoom View (Last {zoom_weeks} Weeks)")

    # Slice test set for zoom period
    zoom_start_date = daily_zoom["date"].min()
    test_zoom = test[test["date"] >= zoom_start_date].copy()
    y_test_zoom = test_zoom["transactions"].values.astype(float)
    test_zoom_dates = test_zoom["date"]

    # Slice SARIMAX predictions and intervals for zoom period
    sarimax_pred_zoom = sarimax_pred[test["date"] >= zoom_start_date]
    lower_zoom = lower[test["date"] >= zoom_start_date]
    upper_zoom = upper[test["date"] >= zoom_start_date]

    # Plot: SARIMAX Forecast Zoom
    fig_sarima_zoom, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(test_zoom_dates, y_test_zoom, label="Observed", color="black", linewidth=2)
    ax.plot(test_zoom_dates, sarimax_pred_zoom, label="SARIMA/SARIMAX Forecast")
    ax.fill_between(test_zoom_dates, lower_zoom, upper_zoom, alpha=0.2, label="95% Prediction Interval")
    ax.set_title(f"SARIMA/SARIMAX Forecast with Prediction Interval (Last {zoom_weeks} Weeks)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Transactions")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.xticks(rotation=28, ha="right")
    fig_sarima_zoom.tight_layout()
    st.pyplot(fig_sarima_zoom, width="stretch")

    # Optional: export zoom figure
    if export_on:
        save_fig(fig_sarima_zoom, out_dir, f"SARIMAX_Forecast_PI_Zoom_{zoom_weeks}w.png")


# ----------------------------
# 6. Model Evaluation
# ----------------------------
st.header("6. Model Evaluation (Test Window)")

rows = [
    {
        "Model": f"Seasonal naive (lag {season_period})",
        "RMSE": rmse(y_test, naive_pred),
        "MAE": mae(y_test, naive_pred),
        "MAPE (%)": mape(y_test, naive_pred),
        "AIC": np.nan,
    },
    {
        "Model": f"ARIMA{arima_order}+exog" if len(exog_cols) else f"ARIMA{arima_order}",
        "RMSE": rmse(y_test, arima_pred),
        "MAE": mae(y_test, arima_pred),
        "MAPE (%)": mape(y_test, arima_pred),
        "AIC": float(arima_res.aic),
    },
    {
        "Model": f"SARIMA{sarima_order}x{sarima_seasonal}+exog"
        if len(exog_cols)
        else f"SARIMA{sarima_order}x{sarima_seasonal}",
        "RMSE": rmse(y_test, sarimax_pred),
        "MAE": mae(y_test, sarimax_pred),
        "MAPE (%)": mape(y_test, sarimax_pred),
        "AIC": float(sarimax_res.aic),
    },
]

if enable_prophet and PROPHET_AVAILABLE and (prophet_pred is not None):
    rows.append(
        {
            "Model": "Prophet+exog" if len(exog_cols) else "Prophet",
            "RMSE": rmse(y_test, prophet_pred),
            "MAE": mae(y_test, prophet_pred),
            "MAPE (%)": mape(y_test, prophet_pred),
            "AIC": np.nan,
        }
    )

metrics_df = pd.DataFrame(rows).sort_values("RMSE")
st.dataframe(metrics_df, width="stretch")

best_model_name = metrics_df.iloc[0]["Model"]
st.success(f"Best model by RMSE: **{best_model_name}**.")

st.subheader("6.1 Metric Comparison Plot")
plot_df = metrics_df.copy()
models = plot_df["Model"].values
rmse_vals = plot_df["RMSE"].values
mae_vals = plot_df["MAE"].values
mape_vals = plot_df["MAPE (%)"].values

x = np.arange(len(models))
w = 0.35

fig_metrics, axm = plt.subplots(figsize=(11, 4.8))
axm.bar(x - w / 2, rmse_vals, w, label="RMSE")
axm.bar(x + w / 2, mae_vals, w, label="MAE")
axm.set_xticks(x)
axm.set_xticklabels(models, rotation=20, ha="right")
axm.set_ylabel("Error (transactions)")
axm.set_title("Model Accuracy Metrics")

axm2 = axm.twinx()
axm2.plot(x, mape_vals, marker="o", linewidth=2, label="MAPE (%)")
axm2.set_ylabel("MAPE (%)")

h1, l1 = axm.get_legend_handles_labels()
h2, l2 = axm2.get_legend_handles_labels()
axm.legend(h1 + h2, l1 + l2, loc="upper right")

fig_metrics.tight_layout()
st.pyplot(fig_metrics, width="stretch")

# ----------------------------
# 7. Forecast Visualization
# ----------------------------
st.header("7. Forecast Visualization")

st.subheader("7.1 Model Comparison (Test Window)")
fig_cmp = plt.figure(figsize=(11, 4.3))
plt.plot(test["date"], y_test, label="Observed")
plt.plot(test["date"], sarimax_pred, label="SARIMA/SARIMAX")
plt.plot(test["date"], arima_pred, label="ARIMA/ARIMAX")
plt.plot(test["date"], naive_pred, label=f"Seasonal naive (lag {season_period})")
if enable_prophet and PROPHET_AVAILABLE and (prophet_pred is not None):
    plt.plot(test["date"], prophet_pred, label="Prophet")
plt.xlabel("Date")
plt.ylabel("Transactions")
plt.title("Observed vs Forecasted Transactions (Test Window)")
plt.legend()
plt.tight_layout()
st.pyplot(fig_cmp, width="stretch")

st.subheader("7.2 SARIMA/SARIMAX Forecast Interval")
fig_pi, ax = plt.subplots(figsize=(11, 4.3))
ax.plot(test["date"], y_test, label="Observed")
ax.plot(test["date"], sarimax_pred, label="Forecast")
ax.fill_between(test["date"], lower, upper, alpha=0.2, label="95% prediction interval")
ax.set_xlabel("Date")
ax.set_ylabel("Transactions")
ax.set_title("Forecast with Prediction Interval (SARIMA/SARIMAX)")
ax.legend()
import matplotlib.dates as mdates

# ... your existing plotting code ...

# Make x-axis show weekly ticks and rotate labels
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=7))  # every 7 days
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=28, ha="right")  # rotate for readability

fig_pi.tight_layout()

fig_pi.tight_layout()
st.pyplot(fig_pi, width="stretch")

if enable_prophet and PROPHET_AVAILABLE and (prophet_pred is not None):
    st.subheader("7.3 Prophet Forecast Interval")
    fig_prophet_pi, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(test["date"], y_test, label="Observed")
    ax.plot(test["date"], prophet_pred, label="Forecast")
    ax.fill_between(
        test["date"],
        prophet_lo,
        prophet_hi,
        alpha=0.2,
        label=f"{int(float(prophet_interval)*100)}% interval",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Transactions")
    ax.set_title("Forecast with Prediction Interval (Prophet)")
    ax.legend()
    fig_prophet_pi.tight_layout()
    st.pyplot(fig_prophet_pi, width="stretch")
    # ----------------------------
# 7B. Zoom Forecast Visualization (Last N Weeks)
# ----------------------------
if daily_zoom is not None and len(daily_zoom) > 0:
    st.header(f"7B. Forecast Zoom View (Last {zoom_weeks} Weeks)")

    # Slice test set for zoom period
    zoom_start_date = daily_zoom["date"].min()
    test_zoom = test[test["date"] >= zoom_start_date].copy()
    y_test_zoom = test_zoom["transactions"].values.astype(float)
    test_zoom_dates = test_zoom["date"]

    # Slice forecasts for zoom period
    naive_pred_zoom = naive_pred[test["date"] >= zoom_start_date]
    arima_pred_zoom = arima_pred[test["date"] >= zoom_start_date]
    sarimax_pred_zoom = sarimax_pred[test["date"] >= zoom_start_date]
    lower_zoom = lower[test["date"] >= zoom_start_date]
    upper_zoom = upper[test["date"] >= zoom_start_date]

    if enable_prophet and PROPHET_AVAILABLE and (prophet_pred is not None):
        prophet_pred_zoom = prophet_pred[test["date"] >= zoom_start_date]
        prophet_lo_zoom = prophet_lo[test["date"] >= zoom_start_date]
        prophet_hi_zoom = prophet_hi[test["date"] >= zoom_start_date]

    # ----------------------------
    # Plot: Model Comparison (Zoom)
    # ----------------------------
    fig_zoom, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(test_zoom_dates, y_test_zoom, label="Observed", color="black", linewidth=2)
    ax.plot(test_zoom_dates, sarimax_pred_zoom, label="SARIMA/SARIMAX")
    ax.plot(test_zoom_dates, arima_pred_zoom, label="ARIMA/ARIMAX")
    ax.plot(test_zoom_dates, naive_pred_zoom, label=f"Seasonal naive (lag {season_period})")
    if enable_prophet and PROPHET_AVAILABLE and (prophet_pred is not None):
        ax.plot(test_zoom_dates, prophet_pred_zoom, label="Prophet")
    ax.set_title(f"Observed vs Forecast (Last {zoom_weeks} Weeks)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Transactions")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.xticks(rotation=28, ha="right")
    fig_zoom.tight_layout()
    st.pyplot(fig_zoom, width="stretch")

    # ----------------------------
    # Plot: SARIMAX Prediction Interval (Zoom)
    # ----------------------------
    fig_pi_zoom, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(test_zoom_dates, y_test_zoom, label="Observed", color="black", linewidth=2)
    ax.plot(test_zoom_dates, sarimax_pred_zoom, label="SARIMA/SARIMAX Forecast")
    ax.fill_between(test_zoom_dates, lower_zoom, upper_zoom, alpha=0.2, label="95% Prediction Interval")
    ax.set_title(f"SARIMA/SARIMAX Forecast with Prediction Interval (Last {zoom_weeks} Weeks)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Transactions")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.xticks(rotation=28, ha="right")
    fig_pi_zoom.tight_layout()
    st.pyplot(fig_pi_zoom, width="stretch")

    # Optional: export zoom figures
    if export_on:
        save_fig(fig_zoom, out_dir, f"Forecast_Comparison_Zoom_{zoom_weeks}w.png")
        save_fig(fig_pi_zoom, out_dir, f"Forecast_PI_SARIMAX_Zoom_{zoom_weeks}w.png")


# ----------------------------
# 8. Residual Diagnostics
# ----------------------------
st.header("8. Residual Diagnostics (SARIMA/SARIMAX)")
st.subheader("8.1 Residual ACF")
resid = pd.Series(sarimax_res.resid).dropna()
fig_res, axr = plt.subplots(figsize=(9, 3.6))
plot_acf(resid, lags=lags, ax=axr)
axr.set_title("Residual Autocorrelation (ACF)")
fig_res.tight_layout()
st.pyplot(fig_res, width="stretch")
st.caption("Residual ACF with reduced structure suggests improved model fit.")

# ----------------------------
# 9. Export
# ----------------------------
st.header("9. Export")

if export_on:
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save figures (PNG)"):
            # Series overview
            save_fig(fig_ts, out_dir, "TimeSeries_Full.png")
            save_fig(fig_hist, out_dir, "Distribution_Full.png")
            save_fig(fig_w, out_dir, "Weekly_Resample.png")
            save_fig(fig_m, out_dir, "Monthly_Resample.png")
            if show_seasonal_profiles:
                save_fig(fig_dow, out_dir, "Profile_Weekday.png")
                save_fig(fig_month, out_dir, "Profile_Month.png")

            # Zoom view
            if daily_zoom is not None and len(daily_zoom) > 0:
                save_fig(
                    fig_timeseries(daily_zoom, title=f"Daily Transactions (Last {zoom_weeks} weeks)"),
                    out_dir,
                    f"TimeSeries_Zoom_{zoom_weeks}w.png",
                )

            # Decomposition + ACF/PACF
            save_fig(fig_dec, out_dir, "Decomposition.png")
            if show_acf_full:
                save_fig(fig_acf_full, out_dir, "ACF_Full.png")
                save_fig(fig_pacf_full, out_dir, "PACF_Full.png")
            if show_acf_zoom and (daily_zoom is not None) and len(daily_zoom) > 30:
                save_fig(fig_acf_z, out_dir, f"ACF_Zoom_{zoom_weeks}w.png")
                save_fig(fig_pacf_z, out_dir, f"PACF_Zoom_{zoom_weeks}w.png")

            # Models
            save_fig(fig_naive, out_dir, "Model_SeasonalNaive.png")
            save_fig(fig_arima, out_dir, "Model_ARIMAX.png")
            save_fig(fig_sarima, out_dir, "Model_SARIMAX_PI.png")
            if enable_prophet and PROPHET_AVAILABLE and (fig_prophet is not None):
                save_fig(fig_prophet, out_dir, "Model_Prophet.png")
                if fig_prophet_components is not None:
                    save_fig(fig_prophet_components, out_dir, "Prophet_Components.png")

            # Evaluation + Forecast + Residuals
            save_fig(fig_metrics, out_dir, "Metrics_Comparison.png")
            save_fig(fig_cmp, out_dir, "Forecast_Comparison.png")
            save_fig(fig_pi, out_dir, "Forecast_PI_SARIMAX.png")
            save_fig(fig_res, out_dir, "Residual_ACF.png")
            st.success(f"Saved figures to: {out_dir}")

    with col2:
        if st.button("Save metrics table (CSV)"):
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "Model_Evaluation_Metrics.csv"
            metrics_df.to_csv(csv_path, index=False)
            st.success(f"Saved: {csv_path}")

    with col3:
        st.write(f"Output folder: {out_dir}")
else:
    st.info("Enable export in the sidebar to save PNG/CSV outputs.")
