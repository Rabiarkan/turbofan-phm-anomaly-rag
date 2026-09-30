"""Checks the environment. Exit code 1 on failure."""

import os
import platform
import sys
from pathlib import Path

# Make `src` importable even if PYTHONPATH is not set (e.g. during postCreateCommand).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ok = True


def check(name, fn):
    global ok
    try:
        print(f"[ok]   {name}: {fn()}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {e}")


def libs():
    import numpy
    import pandas
    import pyarrow
    import sklearn

    return (
        f"numpy {numpy.__version__}, pandas {pandas.__version__}, "
        f"sklearn {sklearn.__version__}, pyarrow {pyarrow.__version__}"
    )


def torch_info():
    import torch

    build = "cuda build" if torch.version.cuda else "cpu build"
    return f"{torch.__version__} ({build}), threads={torch.get_num_threads()}"


def lstm_autoencoder_step():
    # Tiny LSTM autoencoder on random data: forward, backward, loss must drop.
    import torch
    from torch import nn

    torch.manual_seed(0)
    x = torch.randn(16, 30, 14)  # (batch, window, sensors)

    class AE(nn.Module):
        def __init__(self, n_feat=14, hidden=16):
            super().__init__()
            self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
            self.dec = nn.LSTM(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_feat)

        def forward(self, x):
            _, (h, _) = self.enc(x)
            z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
            y, _ = self.dec(z)
            return self.out(y)

    model = AE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), x)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if losses[-1] >= losses[0]:
        raise RuntimeError(f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}")
    return f"loss {losses[0]:.3f} -> {losses[-1]:.3f}"


def paths():
    from src.config import DATA_PROCESSED

    probe = DATA_PROCESSED / ".write_test"
    probe.write_text("x")
    probe.unlink()
    return f"{DATA_PROCESSED} writable"


def secrets():
    names = ["GROQ_API_KEY"]
    return ", ".join(f"{n}={'set' if os.getenv(n) else 'missing'}" for n in names)


print(f"python {platform.python_version()} on {platform.machine()}, cpus={os.cpu_count()}")
check("libraries", libs)
check("torch", torch_info)
check("lstm autoencoder", lstm_autoencoder_step)
check("paths", paths)
print(f"[info] secrets: {secrets()}")  # missing keys are informational, not a failure
sys.exit(0 if ok else 1)
