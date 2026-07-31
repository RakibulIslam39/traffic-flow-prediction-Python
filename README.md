# Traffic Flow Prediction

Short-term traffic flow prediction using deep neural networks. This project trains and
compares three models — **LSTM**, **GRU**, and **Stacked Auto-Encoders (SAEs)** — on
5‑minute interval traffic flow data and evaluates their forecasting accuracy.

## Overview

Given a window of recent traffic flow observations, each model predicts the flow at the
next time step. The pipeline:

1. Loads and scales the raw traffic flow CSV data.
2. Builds sliding windows of the previous `lag` observations (default `lag = 12`, i.e. the
   last hour of 5‑minute readings) as inputs.
3. Trains an LSTM, GRU, or SAEs model.
4. Loads the trained models and compares their predictions against the ground truth using
   standard regression metrics.

## Project structure

```
.
├── main.py            # Load trained models, predict, evaluate and plot results
├── train.py           # Train a model (lstm | gru | saes)
├── data/
│   ├── data.py        # Data loading, scaling and windowing
│   ├── train.csv      # Training data (PeMS 5-min flow)
│   └── test.csv       # Test data
├── model/
│   ├── model.py       # LSTM, GRU and SAEs model definitions
│   ├── lstm.h5        # Pre-trained LSTM weights
│   ├── gru.h5         # Pre-trained GRU weights
│   ├── saes.h5        # Pre-trained SAEs weights
│   └── *.csv          # Per-model training loss history
├── images/            # Model diagrams and result plots
└── szxs.ipynb         # Exploratory notebook
```

## Requirements

- Python 3.6
- TensorFlow 1.5.0 (GPU recommended)
- Keras 2.1.3
- scikit-learn 0.19
- numpy
- pandas
- matplotlib

Install the dependencies with:

```bash
pip install tensorflow-gpu==1.5.0 keras==2.1.3 scikit-learn==0.19 numpy pandas matplotlib
```

> **Note:** This project targets the Keras 1.x/2.x + TensorFlow 1.x era API
> (e.g. `keras.layers.recurrent`, `keras.utils.vis_utils`). Use the pinned versions above
> for the code to run as-is.

## Dataset

Data are obtained from the **Caltrans Performance Measurement System (PeMS)**. Readings are
collected in real time from individual detectors across California's freeway system. The
relevant column used for prediction is `Lane 1 Flow (Veh/5 Minutes)`, resampled at 5‑minute
intervals.

## Training

Run the command below to train a model:

```bash
python train.py --model lstm
```

You can choose `lstm`, `gru`, or `saes` as the `--model` argument. The trained `.h5` weight
file and a `<model> loss.csv` history file are saved to the `model/` folder.

Default training configuration (see `train.py`):

- `lag`: 12
- `batch`: 256
- `epochs`: 600
- optimizer: RMSprop
- loss: MSE

## Evaluation

Run the command below to evaluate the trained models and plot their predictions against the
true flow:

```bash
python main.py
```

This loads `model/lstm.h5`, `model/gru.h5`, and `model/saes.h5`, prints the metrics below for
each model, and displays a comparison plot of true vs. predicted flow over a day.

Reported metrics: MAE, MSE, RMSE, MAPE, R², and explained variance score.

### Example results

| Metrics | MAE  | MSE   | RMSE | MAPE   |  R²    | Explained variance score |
| ------- |:----:|:-----:|:----:|:------:|:------:|:------------------------:|
| LSTM    | 7.21 | 98.05 | 9.90 | 16.56% | 0.9396 | 0.9419 |
| GRU     | 7.20 | 99.32 | 9.97 | 16.78% | 0.9389 | 0.9389 |
| SAEs    | 7.06 | 92.08 | 9.60 | 17.80% | 0.9433 | 0.9442 |

![evaluation](images/eva.png)

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

This project is licensed under the terms of the [MIT License](LICENSE).
