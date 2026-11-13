# Data

`raw/train.csv` and `raw/test.csv` — 5-minute freeway traffic flow from one detector
(Caltrans PeMS). Columns:

| Column | Meaning |
|---|---|
| `5 Minutes` | Timestamp of the 5-minute interval |
| `Lane 1 Flow (Veh/5 Minutes)` | **Target** — vehicles counted in lane 1 |
| `# Lane Points` | Number of lane points aggregated |
| `% Observed` | Share of the interval that was actually observed |

Size: train = 7,776 rows (27 days x 288), test = 4,320 rows (15 days x 288).

Known issues (to be handled in Phase 3): the timestamp column is read as plain text and its
date format/continuity has not been verified; a handful of zero readings (6 in train) may be
missing data rather than real zero flow.
<<<<<<< HEAD

## PeMS08 (optional, step 4)

`pems08/PEMS08.npz` is a public benchmark (170 sensors x 62 days x [flow, occupancy, speed],
5-minute steps). It is not committed; download it with `scripts/download_pems08.sh`. The file
has no timestamps: `scripts/step4_experiment.py` assumes it starts on Friday 1 July 2016, which
matches its weekly pattern (lower weekends, a low Monday 4 July).
=======
>>>>>>> origin/main
