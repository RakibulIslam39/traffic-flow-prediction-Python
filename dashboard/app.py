"""
Traffic flow forecast dashboard.

    streamlit run dashboard/app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from traffic_flow.data import load_series  # noqa: E402
from traffic_flow.forecast import Forecaster  # noqa: E402

TRAIN = os.path.join(ROOT, "data", "raw", "train.csv")
TEST = os.path.join(ROOT, "data", "raw", "test.csv")
SAVED = os.path.join(ROOT, "artifacts", "forecaster.joblib")
KERAS_DIR = os.path.join(ROOT, "artifacts", "keras3")

st.set_page_config(page_title="Traffic flow forecast", layout="wide")


@st.cache_resource
def get_forecaster():
    return Forecaster.load_or_fit(SAVED, TRAIN)


@st.cache_data
def get_series():
    times, flow = load_series(TEST)
    return pd.DataFrame({"time": pd.to_datetime(times), "flow": flow})


@st.cache_data
def get_walk_forward(horizon):
    return get_forecaster().predict_series(TEST, horizon)


@st.cache_data
def get_backtest():
    return get_forecaster().backtest(TEST)


@st.cache_data
def get_neural_predictions():
    """5-minute predictions of the saved LSTM / GRU / SAEs models (needs TensorFlow)."""
    from traffic_flow.data import make_datasets
    from traffic_flow.evaluate import LABELS, load_models, predict_all

    loaded, meta = load_models(KERAS_DIR)
    y_true, preds = predict_all(loaded, TRAIN, TEST, horizon=meta["horizon"],
                                features=meta["features"])
    times = make_datasets(TRAIN, TEST, 12, val_days=0, shuffle=False,
                          horizon=meta["horizon"])["test_times"]
    out = pd.DataFrame({"time": times, "actual": y_true})
    for name, p in preds.items():
        out[LABELS[name]] = p
    return out, meta


f = get_forecaster()
series = get_series()
days = sorted(series["time"].dt.date.unique())

st.title("Traffic flow forecast")
st.caption("Model: %s. Trained on %d days (%s to %s). The days shown here are the test set, "
           "which the model has never seen."
           % (f.info.get("model", "gradient boosting"), f.info.get("train_days", 0),
              str(f.info.get("train_start", ""))[:10], str(f.info.get("train_end", ""))[:10]))

with st.sidebar:
    st.header("Controls")
    day = st.selectbox("Day", days, format_func=lambda d: d.strftime("%a %d %b %Y"))
    horizon_min = st.selectbox("Forecast horizon", [5 * h for h in f.horizons],
                               format_func=lambda m: "%d minutes ahead" % m)
    moment = st.slider("Time of day (for the single forecast)", 1.0, 23.0, 8.0, 0.25,
                       format="%.2f h")
horizon = horizon_min // 5

# ------------------------------------------------------------------ single forecast
st.subheader("Forecast from one moment")
now = pd.Timestamp(day) + pd.Timedelta(hours=float(moment))
recent = series[(series["time"] <= now) & (series["time"] > now - pd.Timedelta(minutes=60))]
if len(recent) != f.lags:
    st.warning("There are not 12 consecutive readings before %s on this day." % now.strftime("%H:%M"))
else:
    forecasts = pd.DataFrame(f.predict(recent["flow"].values, now))
    forecasts["time"] = pd.to_datetime(forecasts["time"])
    forecasts = forecasts.merge(series.rename(columns={"flow": "actual"}), on="time", how="left")
    cols = st.columns(len(forecasts) + 1)
    cols[0].metric("Now (%s)" % now.strftime("%H:%M"), "%.0f" % recent["flow"].iloc[-1])
    for c, (_, r) in zip(cols[1:], forecasts.iterrows()):
        delta = None if pd.isna(r["actual"]) else "actual %.0f" % r["actual"]
        c.metric("+%d min" % r["horizon_minutes"], "%.0f" % r["flow"], delta, delta_color="off")
    st.caption("Vehicles per 5 minutes. The forecast uses only the 12 readings up to %s and the "
               "time of day; the actual values are shown for comparison." % now.strftime("%H:%M"))

# ------------------------------------------------------------------ day view
st.subheader("Whole day, %d minutes ahead" % horizon_min)
walk = get_walk_forward(horizon)
day_rows = walk[walk["time"].dt.date == day].set_index("time")[["actual", "forecast"]]
if len(day_rows):
    st.line_chart(day_rows)
    err = (day_rows["actual"] - day_rows["forecast"]).abs().mean()
    st.caption("Each point was forecast %d minutes earlier. Mean absolute error on this day: "
               "%.1f vehicles per 5 minutes." % (horizon_min, err))
else:
    st.info("No forecasts are possible on this day at this horizon.")

# ------------------------------------------------------------------ accuracy
st.subheader("Accuracy on the whole test set")
bt = get_backtest().rename(columns={
    "horizon_minutes": "Minutes ahead", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)",
    "persistence_mae": "MAE if you just repeat the last reading", "n": "Forecasts"})
st.dataframe(bt.round(2), hide_index=True)

# ------------------------------------------------------------------ neural comparison
st.subheader("Compare with the neural models")
if st.checkbox("Load LSTM / GRU / SAEs (starts TensorFlow, takes a few seconds)"):
    try:
        neural, meta = get_neural_predictions()
    except Exception as err:  # TensorFlow missing, no saved models, ...
        st.error("Could not load the neural models: %s" % err)
    else:
        shown = neural[neural["time"].dt.date == day].set_index("time")
        gb = get_walk_forward(meta["horizon"])
        shown = shown.join(gb.set_index("time")["forecast"].rename("GradBoost"), how="left")
        st.line_chart(shown)
        errors = shown.drop(columns="actual").sub(shown["actual"], axis=0).abs().mean()
        st.caption("%d minutes ahead. Mean absolute error on this day: %s."
                   % (5 * meta["horizon"],
                      ", ".join("%s %.1f" % (k, v) for k, v in errors.items())))
