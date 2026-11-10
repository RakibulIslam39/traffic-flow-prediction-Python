# Traffic Flow Prediction

Short-term traffic flow prediction with deep neural networks. Three models — **LSTM**, **GRU**
and **Stacked Auto-Encoders (SAEs)** — are trained on 5-minute freeway traffic flow data and
compared on their next-step forecasting accuracy.

> **Status: Phase 2 (modernised stack).** This started as a university capstone project.
> The code now runs on Keras 3 / TensorFlow 2. The original Keras 2.1 weights still load and
> reproduce the original numbers, but models retrained with the new code are **not yet as
> accurate** — see [Results](#results-as-reported-by-the-original-project).

## Attribution

The model code is based on an open-source traffic flow prediction project released under the
MIT License (Copyright (c) 2018 Larry — see [LICENSE](LICENSE)). Keep this notice in any
redistribution. Data come from the Caltrans Performance Measurement System (PeMS).

## How it works

1. Load the `Lane 1 Flow (Veh/5 Minutes)` series and scale it to [0, 1] (scaler fit on train only).
2. Build sliding windows: the previous `lag` readings (default 12 = 1 hour) predict the next one.
3. Train an LSTM, GRU or SAEs model (MSE loss, RMSprop, batch 256, 600 epochs).
4. Load trained models, predict on the test set and compare with the ground truth.

## Project structure

```
.
├── src/traffic_flow/
│   ├── data.py         # loading, scaling, windowing
│   ├── models.py       # LSTM, GRU, SAEs definitions (Keras 3)
│   ├── legacy.py       # load original Keras 2.1 .h5 weights into the new models
│   ├── metrics.py      # MAE, MSE, RMSE, MAPE, R², explained variance
│   ├── train.py        # python -m traffic_flow.train --model lstm|gru|saes|all
│   └── evaluate.py     # python -m traffic_flow.evaluate [--legacy]
├── data/
│   ├── raw/            # train.csv, test.csv  (see data/README.md)
│   └── README.md
├── artifacts/
│   └── legacy/         # original trained weights (.h5) and loss histories
├── docs/images/        # model diagrams and result plot
├── tests/              # data, metrics, models, legacy weights, training smoke tests
├── requirements.txt          # TensorFlow 2 + Keras 3
├── requirements-dev.txt      # + pytest
└── pyproject.toml
```

## Quick start

Requires Python 3.10+.

```bash
make install       # or: pip install -r requirements.txt && pip install -e .

# Train (600 epochs; use --epochs 2 for a quick check)
python -m traffic_flow.train --model lstm        # or gru, saes, all

# Evaluate models trained by `train.py` (default folder: artifacts/)
python -m traffic_flow.evaluate

# Evaluate the Keras 3 models retrained in Phase 2 (kept in artifacts/keras3/)
python -m traffic_flow.evaluate --weights-dir artifacts/keras3

# Evaluate the original Keras 2.1 weights (artifacts/legacy/*.h5)
python -m traffic_flow.evaluate --legacy --legacy-windows
```

Evaluation writes `metrics.csv` and `comparison.png` to `artifacts/eval/` (or `--out-dir`).
The plot shows the first full day of the test set.

### Makefile shortcuts

```bash
make install        # dependencies + editable install
make test           # unit and regression tests (~20 s)
make smoke          # whole pipeline on 2 epochs in a temp dir, with output checks (~1 min)
make train MODEL=lstm EPOCHS=600 SEED=42
make evaluate
make reproduce      # check that the original weights still give the README numbers
make help           # list everything
```

CI (`.github/workflows/ci.yml`) runs `make test` and `make smoke` on Python 3.11 and 3.12.

The original Keras 2.1 `.h5` files cannot be loaded with `keras.saving.load_model`.
`traffic_flow.legacy` rebuilds the architecture with the original `hard_sigmoid` gates and
copies the weights across.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

`tests/test_legacy_weights.py` checks that the original weights reproduce the MAE and MSE
in the original README. The rest run quickly on tiny synthetic data.

```
```

## Results

**Original weights** (`--legacy`, reproduce the original README exactly):

| Model | MAE | MSE | RMSE | MAPE | R² | Explained var. |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| LSTM | 7.216 | 98.05 | 9.90 | 16.59% | 0.9396 | 0.9419 |
| GRU  | 7.202 | 99.32 | 9.97 | 16.78% | 0.9389 | 0.9389 |
| SAEs | 7.058 | 92.08 | 9.60 | 17.80% | 0.9433 | 0.9443 |

**Retrained with the Keras 3 code** (`train.py --model all --seed 42`, 600 epochs, `artifacts/keras3/`).
These were trained with the **old** windowing (see below), so they are not yet comparable with the
corrected evaluation:

| Model | MAE | MSE | RMSE | MAPE | R² | Explained var. |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| LSTM | 7.688 | 105.62 | 10.28 | 24.43% | 0.9350 | 0.9396 |
| GRU  | 7.776 | 107.98 | 10.39 | 24.89% | 0.9335 | 0.9385 |
| SAEs | 8.065 | 115.56 | 10.75 | 25.13% | 0.9289 | 0.9355 |

The retrained models are about 6–14% worse on MAE than the original weights, and they do
not reproduce the original numbers. Single seed only. The likely cause is that `train.py`
uses the modern `sigmoid` gate for the recurrent layers, not the legacy `hard_sigmoid`
gate the original weights were trained with. This is a hypothesis that has not been tested
yet.

**Corrected windowing** (current default): windows never span a gap in the timestamps. The
original weights give MAE 7.28 / 7.26 / 7.12 (LSTM / GRU / SAEs) with it, compared with 7.22 /
7.20 / 7.06 under the original windowing. Retraining with the corrected windows is the next step.

**Baselines** (same test windows):

| Baseline | MAE | RMSE | MAPE |
| --- | :---: | :---: | :---: |
| Persistence (previous 5-min value) | 8.34 | 11.31 | 20.56% |
| Seasonal naive (same time yesterday) | 10.21 | 14.06 | 24.43% |

The neural models improve MAE over plain persistence by only 3–14%. The retrained models
beat persistence on MAE but not on MAPE. A proper baseline comparison is part of Phase 3.

![evaluation](docs/images/eva.png)

**Baseline check (computed on `data/raw/test.csv`, same windows as the models):**

| Baseline | MAE | RMSE | MAPE |
| --- | :---: | :---: | :---: |
| Persistence (predict previous 5-min value) | 8.34 | 11.31 | 20.56% |
| Seasonal naive (same time yesterday) | 10.21 | 14.06 | 24.43% |

The neural models improve MAE over plain persistence by only about 14%. A proper baseline
comparison is part of Phase 3.

![evaluation](docs/images/eva.png)

## Issues found and fixed in the testing pass

- **Windows crossed timestamp gaps.** The train and test files contain whole missing days
  (10 gaps in train, 5 in test). Windows spanning a gap were treated as consecutive readings.
  Fixed; `process_data(..., contiguous=False)` keeps the old behaviour for reproduction.
- **Plot time axis was off by one hour** and had a hard-coded date. It now plots the first full
  day of the test set, selected by timestamp.
- **Missing or wrong paths gave raw Keras/pandas tracebacks.** Now they give a short message
  that says what is missing and how to fix it.
- **TensorFlow C++ logs** flooded the console. Silenced by default (`TF_CPP_MIN_LOG_LEVEL`).
- **Smoke-test fixture** wrote CSVs without timestamps; the tests now use the real file format.
- **Reproducibility** checked: same seed gives identical weights, different seeds do not.

## Known limitations

- Single detector, single lane, univariate input, one-step-ahead only.
- Training windows are shuffled before `validation_split`, so the validation loss leaks
  (overlapping windows) and cannot be trusted; no early stopping.
- Missing values are filled with 0. The raw files contain no NaNs, and the few zero readings
  are mostly genuine low-traffic hours, so this has limited effect.
- The "SAEs" layers are trained to predict the target, not to reconstruct the input.
- Small dataset (about 27 training days, 15 test days).

## Roadmap

- [x] **Phase 1** — repo cleanup, structure, attribution, tests for the data pipeline
- [~] **Phase 2** — modern stack (Keras 3 / TF 2) done; retrained models do not yet match the original accuracy (see Results)
- [x] **Testing pass** — end-to-end smoke test, Makefile, CI, bug fixes (see above)
- [ ] **Phase 3** — chronological split, proper validation, baselines, multi-seed evaluation
- [ ] **Phase 4** — time features, multi-step horizons, new models
- [ ] **Phase 5** — CLI/config, API + dashboard, Docker, CI
- [ ] **Phase 6** — documentation and report

## References

```
@article{SAEs,
  title={Traffic Flow Prediction With Big Data: A Deep Learning Approach},
  author={Y Lv, Y Duan, W Kang, Z Li, FY Wang},
  journal={IEEE Transactions on Intelligent Transportation Systems, 2015, 16(2):865-873},
  year={2015}
}

@article{RNN,
  title={Using LSTM and GRU neural network methods for traffic flow prediction},
  author={R Fu, Z Zhang, L Li},
  journal={Chinese Association of Automation, 2017:324-328},
  year={2017}
}
```

## License

MIT — see [LICENSE](LICENSE).
