#!/usr/bin/env bash
# End-to-end smoke test of the whole pipeline on tiny settings.
#   train (all models, 2 epochs) -> evaluate -> legacy evaluate -> check every output.
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

step "check metrics"
"$PY" - "$WORK" <<'PYEOF'
import math, sys, pandas as pd
work = sys.argv[1]
checks = {
    "eval/metrics.csv": 3,     # one row per model
    "legacy/metrics.csv": 3,
}
for rel, rows in checks.items():
    df = pd.read_csv(f"{work}/{rel}", index_col=0)
    assert len(df) == rows, f"{rel}: expected {rows} rows, got {len(df)}"
    assert all(math.isfinite(v) for v in df.values.ravel()), f"{rel}: non-finite metric"
    assert (df["mae"] >= 0).all(), f"{rel}: negative MAE"
    print(f"{rel}: ok")
legacy = pd.read_csv(f"{work}/legacy/metrics.csv", index_col=0)
# the legacy weights must still reproduce the original README MAE (+/- 0.01)
expected = {"LSTM": 7.21, "GRU": 7.20, "SAEs": 7.06}
for name, mae in expected.items():
    assert abs(legacy.loc[name, "mae"] - mae) < 0.01, f"{name} legacy MAE {legacy.loc[name, 'mae']:.3f} != {mae}"
print("legacy MAE matches original README: ok")
PYEOF

printf '\nSMOKE TEST PASSED\n'
