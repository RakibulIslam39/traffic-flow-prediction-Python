"""Short-term traffic flow prediction (LSTM / GRU / SAEs)."""
import os

# Hide TensorFlow's C++ startup chatter unless the user has set these explicitly.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

__version__ = "0.2.0"
