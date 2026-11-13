import json
import os

import numpy as np
import pandas as pd
import pytest

from traffic_flow import predict as predict_cli
from traffic_flow.forecast import Forecaster

ATTR = "Lane 1 Flow (Veh/5 Minutes)"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _csv(path, days, start="2016-01-04", drop_last_hour_gap=False):
    """Daily sine pattern, so the time features carry real information."""
    t = pd.date_range(start, periods=days * 288, freq="5min")
    flow = 60 + 50 * np.sin(2 * np.pi * (t.hour * 60 + t.minute) / 1440)
    df = pd.DataFrame({"5 Minutes": t, ATTR: np.round(flow, 2)})
    if drop_last_hour_gap:
        df = df.drop(df.index[-5])
    df["5 Minutes"] = df["5 Minutes"].dt.strftime("%d/%m/%Y %H:%M")
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("fc")
    _csv(d / "train.csv", 6)
    _csv(d / "test.csv", 2, start="2016-02-01")
    return str(d / "train.csv"), str(d / "test.csv"), d


@pytest.fixture(scope="module")
def forecaster(paths):
    return Forecaster.fit(paths[0])


def test_predict_returns_one_forecast_per_horizon_with_right_times(forecaster):
    out = forecaster.predict([60.0] * 12, "2016-02-01 08:00")
    assert [r["horizon_minutes"] for r in out] == [5, 15, 30, 60]
    assert [r["time"] for r in out] == ["2016-02-01 08:05", "2016-02-01 08:15",
                                        "2016-02-01 08:30", "2016-02-01 09:00"]
    assert all(r["flow"] >= 0 for r in out)


def test_forecasts_follow_the_daily_pattern(forecaster, paths):
    bt = forecaster.backtest(paths[1])
    # the series is a pure daily cycle: with time features the error stays small at every
    # horizon, while repeating the last reading gets worse as the horizon grows
    assert (bt["mae"] < 2.0).all()
    assert bt["persistence_mae"].is_monotonic_increasing
    assert bt["mae"].iloc[-1] < bt["persistence_mae"].iloc[-1] / 3


@pytest.mark.parametrize("bad", [[1.0] * 11, [1.0] * 13, [1.0] * 11 + [float("nan")],
                                 [1.0] * 11 + [-5.0]])
def test_bad_readings_are_rejected(forecaster, bad):
    with pytest.raises(ValueError):
        forecaster.predict(bad, "2016-02-01 08:00")


def test_save_and_load_give_identical_forecasts(forecaster, paths):
    path = forecaster.save(str(paths[2] / "f.joblib"))
    again = Forecaster.load(path)
    readings = list(np.linspace(40, 90, 12))
    assert again.predict(readings, "2016-02-01 10:00") == forecaster.predict(readings, "2016-02-01 10:00")
    assert again.info["train_days"] == 6


def test_load_or_fit_creates_the_file_once(paths):
    path = str(paths[2] / "auto.joblib")
    assert not os.path.exists(path)
    Forecaster.load_or_fit(path, paths[0])
    assert os.path.exists(path)
    stamp = os.path.getmtime(path)
    Forecaster.load_or_fit(path, paths[0])
    assert os.path.getmtime(path) == stamp


def test_predict_from_csv_uses_last_rows_and_rejects_gaps(forecaster, paths):
    out = forecaster.predict_from_csv(paths[1])
    assert out[0]["time"] == "2016-02-03 00:00"       # 5 minutes after the last row (23:55)
    gapped = str(paths[2] / "gap.csv")
    _csv(gapped, 1, start="2016-03-01", drop_last_hour_gap=True)
    with pytest.raises(ValueError, match="not 5 minutes apart"):
        forecaster.predict_from_csv(gapped)


def test_predict_series_matches_single_predictions(forecaster, paths):
    s = forecaster.predict_series(paths[1], 3)
    row = s.iloc[100]
    t, flow = pd.read_csv(paths[1])["5 Minutes"], pd.read_csv(paths[1])[ATTR]
    times = pd.to_datetime(t, format="%d/%m/%Y %H:%M")
    i = int(np.flatnonzero(times == row["time"])[0])
    single = forecaster.predict(flow.values[i - 3 + 1 - 12: i - 3 + 1], times[i - 3])
    assert single[1]["horizon_minutes"] == 15 and single[1]["time"] == row["time"].strftime("%Y-%m-%d %H:%M")
    assert single[1]["flow"] == pytest.approx(row["forecast"], abs=0.01)


def test_cli_table_json_and_errors(paths, capsys):
    fpath = str(paths[2] / "cli.joblib")
    common = ["--forecaster", fpath, "--train-csv", paths[0]]
    assert predict_cli.main(common + ["--csv", paths[1]]) == 0
    assert "60 min" in capsys.readouterr().out
    assert predict_cli.main(common + ["--json", "--time", "2016-02-01 08:00", "--readings"]
                            + ["60"] * 12) == 0
    assert len(json.loads(capsys.readouterr().out)) == 4
    with pytest.raises(SystemExit) as e:
        predict_cli.main(common + ["--time", "2016-02-01 08:00", "--readings", "1", "2"])
    assert e.value.code == 2
    assert predict_cli.main(common + ["--backtest", "--test-csv", paths[1]]) == 0
    assert "persistence_mae" in capsys.readouterr().out


def test_real_data_backtest_beats_persistence_at_every_horizon():
    f = Forecaster.fit(os.path.join(ROOT, "data", "raw", "train.csv"))
    bt = f.backtest(os.path.join(ROOT, "data", "raw", "test.csv"))
    assert (bt["mae"] < bt["persistence_mae"]).all()
    assert bt["mae"].iloc[0] == pytest.approx(6.54, abs=0.15)     # 5-minute MAE
    assert bt["mae"].iloc[-1] == pytest.approx(7.14, abs=0.15)    # 60-minute MAE
