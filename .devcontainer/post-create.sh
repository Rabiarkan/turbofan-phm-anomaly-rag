#!/usr/bin/env bash
# Runs once when the codespace is created.
set -euo pipefail

# Codespaces can install Git LFS hooks at clone time, but git-lfs is not in this
# image and the repo does not use LFS, so the pre-push hook would block git push.
if ! command -v git-lfs >/dev/null && ! grep -qs "filter=lfs" .gitattributes; then
  for h in pre-push post-checkout post-commit post-merge; do
    if grep -qs "git lfs" ".git/hooks/$h"; then rm ".git/hooks/$h"; fi
  done
fi

python -m pip install --upgrade pip

# CPU-only PyTorch, installed before requirements.txt (see README).
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt
python -m ipykernel install --user --name phm --display-name "Python (phm)"

# Quick check until scripts/smoke_test.py exists
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.version.cuda)"