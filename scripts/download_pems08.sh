#!/usr/bin/env bash
# Download the PeMS08 benchmark (170 sensors, 62 days of 5-minute data, 18.5 MB).
# The file is not kept in this repository; it is fetched from the ASTGNN authors' repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data/pems08"
curl -L --fail -o "$ROOT/data/pems08/PEMS08.npz" \
  https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data/PEMS08/PEMS08.npz
ls -la "$ROOT/data/pems08/PEMS08.npz"
