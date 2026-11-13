import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from traffic_flow import api  # noqa: E402
from tests.test_forecast import _csv  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    _csv(tmp_path / "train.csv", 6)
    monkeypatch.setenv("TRAFFIC_TRAIN_CSV", str(tmp_path / "train.csv"))
    monkeypatch.setenv("TRAFFIC_FORECASTER", str(tmp_path / "f.joblib"))
    monkeypatch.setattr(api, "_forecaster", None)
    return TestClient(api.app)


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model").json()
    assert info["lags"] == 12 and info["horizons_minutes"] == [5, 15, 30, 60]
    assert info["train_days"] == 6


def test_predict_returns_forecasts(client):
    r = client.post("/predict", json={"readings": [60.0] * 12,
                                      "last_timestamp": "2016-02-01T08:00:00"})
    assert r.status_code == 200
    f = r.json()["forecasts"]
    assert [x["horizon_minutes"] for x in f] == [5, 15, 30, 60]
    assert f[0]["time"] == "2016-02-01 08:05" and all(x["flow"] >= 0 for x in f)


@pytest.mark.parametrize("body", [
    {"readings": [60.0] * 5, "last_timestamp": "2016-02-01T08:00:00"},      # too few
    {"readings": [60.0] * 11 + [-1.0], "last_timestamp": "2016-02-01T08:00:00"},  # negative
    {"readings": [60.0] * 12},                                               # no timestamp
    {"readings": ["x"] * 12, "last_timestamp": "2016-02-01T08:00:00"},      # not numbers
    {"readings": [60.0] * 12, "last_timestamp": "yesterday"},               # bad timestamp
])
def test_bad_requests_get_422(client, body):
    assert client.post("/predict", json=body).status_code == 422
