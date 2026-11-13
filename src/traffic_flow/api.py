"""
HTTP API for the forecaster.

    uvicorn traffic_flow.api:app --reload          # then open http://127.0.0.1:8000/docs

Endpoints:  GET /health   GET /model   POST /predict
"""
import os
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from traffic_flow.forecast import DEFAULT_PATH, Forecaster

app = FastAPI(title="Traffic flow forecast", version="1.0")
_forecaster = None


def get_forecaster():
    """Load (or fit) the forecaster once per process."""
    global _forecaster
    if _forecaster is None:
        _forecaster = Forecaster.load_or_fit(
            os.environ.get("TRAFFIC_FORECASTER", DEFAULT_PATH),
            os.environ.get("TRAFFIC_TRAIN_CSV", "data/raw/train.csv"))
    return _forecaster


class PredictRequest(BaseModel):
    readings: List[float] = Field(
        ..., description="The 12 most recent 5-minute flow readings, oldest first.")
    last_timestamp: datetime = Field(..., description="When the last reading was measured.")


class Forecast(BaseModel):
    horizon_minutes: int
    time: str
    flow: float


class PredictResponse(BaseModel):
    forecasts: List[Forecast]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model():
    f = get_forecaster()
    return {"lags": f.lags, "horizons_minutes": [5 * h for h in f.horizons], **f.info}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return {"forecasts": get_forecaster().predict(req.readings, req.last_timestamp)}
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
