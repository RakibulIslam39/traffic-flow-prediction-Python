#!/usr/bin/env bash
# End-to-end smoke test of the whole pipeline on tiny settings.
#   train (all models, 2 epochs) -> evaluate -> legacy evaluate -> multi-seed experiment
#   -> check every output.
# Runs in a temporary directory so the repository is not touched.
# Usage: scripts/smoke_test.sh [--keep]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
KEEP="${1:-}"
export TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"

cleanup() { if [[ "$KEEP" != "--keep" ]]; then rm -rf "$WORK"; else echo "kept: $WORK"; fi; }
trap cleanup EXIT

step() { printf '\n==> %s\n' "$*"; }
fail() { echo "SMOKE TEST FAILED: $*" >&2; exit 1; }

step "train all models (2 epochs)"
"$PY" -m traffic_flow.train --model all --epochs 2 --out-dir "$WORK/models" --quiet \
  --train-csv "$ROOT/data/raw/train.csv" --test-csv "$ROOT/data/raw/test.csv"
for f in lstm.keras gru.keras saes.keras lstm_loss.csv gru_loss.csv saes_loss.csv; do
  [[ -s "$WORK/models/$f" ]] || fail "missing output $f"
done

step "evaluate freshly trained models"
"$PY" -m traffic_flow.evaluate --weights-dir "$WORK/models" --out-dir "$WORK/eval" \
  --train-csv "$ROOT/data/raw/train.csv" --test-csv "$ROOT/data/raw/test.csv"
[[ -s "$WORK/eval/metrics.csv" && -s "$WORK/eval/comparison.png" ]] || fail "evaluation outputs missing"

step "evaluate the original legacy weights"
"$PY" -m traffic_flow.evaluate --legacy --legacy-windows --out-dir "$WORK/legacy" \
  --train-csv "$ROOT/data/raw/train.csv" --test-csv "$ROOT/data/raw/test.csv" >/dev/null

step "15-minute forecast with time features (1 model, 1 epoch) and its evaluation"
"$PY" -m traffic_flow.train --model gru --epochs 1 --horizon 3 --features \
  --out-dir "$WORK/h3" --quiet \
  --train-csv "$ROOT/data/raw/train.csv" --test-csv "$ROOT/data/raw/test.csv"
"$PY" -m traffic_flow.evaluate --weights-dir "$WORK/h3" --out-dir "$WORK/h3eval" \
  --train-csv "$ROOT/data/raw/train.csv" --test-csv "$ROOT/data/raw/test.csv" > "$WORK/h3eval.log"
grep -q "forecast horizon: 15 min, time features: on" "$WORK/h3eval.log" \
  || fail "horizon/features not picked up by evaluate"

step "forecaster: fit, backtest and a JSON forecast"
"$PY" -m traffic_flow.predict --forecaster "$WORK/forecaster.joblib" --backtest \
  --train-csv "$ROOT/data/raw/train.csv" --test-csv "$ROOT/data/raw/test.csv" > "$WORK/backtest.log"
grep -q "persistence_mae" "$WORK/backtest.log" || fail "backtest table missing"
"$PY" -m traffic_flow.predict --forecaster "$WORK/forecaster.joblib" --json \
  --csv "$ROOT/data/raw/test.csv" > "$WORK/forecast.json"
"$PY" -c "import json,sys; f=json.load(open(sys.argv[1])); assert [x['horizon_minutes'] for x in f]==[5,15,30,60], f" \
  "$WORK/forecast.json" || fail "forecast JSON is not the expected shape"

step "multi-seed experiment (1 model, 2 seeds, 1 epoch)"
"$PY" -m traffic_flow.experiment --models gru --seeds 1 2 --epochs 1 --out-dir "$WORK/experiment" \
  --train-csv "$ROOT/data/raw/train.csv" --test-csv "$ROOT/data/raw/test.csv" >/dev/null
for f in runs.csv summary_default.csv baselines_default.csv; do
  [[ -s "$WORK/experiment/$f" ]] || fail "missing experiment output $f"
done

step "check metrics"
"$PY" - "$WORK" <<'PYEOF'
import math, sys, pandas as pd
work = sys.argv[1]
checks = {
    "eval/metrics.csv": 6,     # 3 neural models + 3 baselines
    "legacy/metrics.csv": 6,
    "experiment/runs.csv": 2,  # 1 model x 2 seeds
    "h3eval/metrics.csv": 4,   # 1 neural model + 3 baselines
}
for rel, rows in checks.items():
    df = pd.read_csv(f"{work}/{rel}", index_col=0)
    assert len(df) == rows, f"{rel}: expected {rows} rows, got {len(df)}"
    num = df.select_dtypes("number")
    assert all(math.isfinite(v) for v in num.values.ravel()), f"{rel}: non-finite metric"
    mae = "mae" if "mae" in df.columns else "test_mae"
    assert (df[mae] >= 0).all(), f"{rel}: negative MAE"
    print(f"{rel}: ok")
legacy = pd.read_csv(f"{work}/legacy/metrics.csv", index_col=0)
# the legacy weights must still reproduce the original README MAE (+/- 0.01)
expected = {"LSTM": 7.21, "GRU": 7.20, "SAEs": 7.06}
for name, mae in expected.items():
    assert abs(legacy.loc[name, "mae"] - mae) < 0.01, f"{name} legacy MAE {legacy.loc[name, 'mae']:.3f} != {mae}"
print("legacy MAE matches original README: ok")
PYEOF

printf '\nSMOKE TEST PASSED\n'
