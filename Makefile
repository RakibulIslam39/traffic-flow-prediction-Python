# Common tasks. Run `make help` for the list.
PY ?= python3
EPOCHS ?= 600
SEED ?= 42
MODEL ?= all

export TF_CPP_MIN_LOG_LEVEL := 2
export TF_ENABLE_ONEDNN_OPTS := 0

.PHONY: help install install-dev test smoke train evaluate legacy-eval reproduce ci clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## install the package and runtime dependencies
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

install-dev: install  ## also install test tools
	$(PY) -m pip install -r requirements-dev.txt

test:  ## unit and regression tests (about 20 s)
	PYTHONPATH=src $(PY) -m pytest -q

smoke:  ## full pipeline on 2 epochs in a temp dir, with output checks (about 1 min)
	PYTHON=$(PY) scripts/smoke_test.sh

train:  ## train models: make train MODEL=lstm|gru|saes|all EPOCHS=600 SEED=42
	PYTHONPATH=src $(PY) -m traffic_flow.train --model $(MODEL) --epochs $(EPOCHS) --seed $(SEED)

evaluate:  ## evaluate models trained into artifacts/
	PYTHONPATH=src $(PY) -m traffic_flow.evaluate

legacy-eval:  ## evaluate the original Keras 2.1 weights (original windowing = README numbers)
	PYTHONPATH=src $(PY) -m traffic_flow.evaluate --legacy --legacy-windows --out-dir artifacts/eval-legacy

reproduce: legacy-eval  ## check that the original weights reproduce the README
	@PYTHONPATH=src $(PY) -c "import pandas as pd; df=pd.read_csv('artifacts/eval-legacy/metrics.csv', index_col=0); print(df[['mae','mse','mape']].round(2)); exp={'LSTM':7.21,'GRU':7.20,'SAEs':7.06}; bad=[k for k,v in exp.items() if abs(df.loc[k,'mae']-v)>=0.01]; raise SystemExit('MISMATCH: %s' % bad if bad else 0)" && echo "README numbers reproduced"

ci: test smoke  ## what CI runs

clean:  ## remove caches and evaluation output (keeps trained models and legacy weights)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache artifacts/eval artifacts/eval-legacy src/*.egg-info
