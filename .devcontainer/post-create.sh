#!/usr/bin/env bash
# Runs once when the codespace is created.
set -euo pipefail

python -m pip install --upgrade pip

# CPU-only PyTorch, installed before requirements.txt (see README).
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt
python -m ipykernel install --user --name phm --display-name "Python (phm)"

# Quick check until scripts/smoke_test.py exists
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.version.cuda)"