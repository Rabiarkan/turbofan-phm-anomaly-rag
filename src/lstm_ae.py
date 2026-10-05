"""LSTM autoencoder anomaly score: reconstruction error of a window of cycles.

Same interface as the baselines: fit(X, X_val) and score(X), where X comes from
views.window_view (n_windows, WINDOW * n_sensors). Trained on healthy windows only;
early stopping uses val-healthy reconstruction loss, never failure information.
"""

import copy

import numpy as np
import torch
from torch import nn

from src.config import (
    FEATURE_SENSORS,
    LSTM_BATCH,
    LSTM_HIDDEN,
    LSTM_LR,
    LSTM_MAX_EPOCHS,
    LSTM_PATIENCE,
)


class _Net(nn.Module):
    def __init__(self, n_feat: int, hidden: int):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.enc(x)  # h[-1]: summary of the window
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)  # one copy per time step
        y, _ = self.dec(z)
        return self.out(y)


class LSTMAE:
    uses_val = True  # runner passes val-healthy windows for early stopping

    def __init__(self, seed: int = 0, hidden: int = LSTM_HIDDEN):
        self.seed, self.hidden = seed, hidden
        self.n_feat = len(FEATURE_SENSORS)
        self.history: list[tuple[float, float]] = []

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32).reshape(len(X), -1, self.n_feat)

    def fit(self, X: np.ndarray, X_val: np.ndarray) -> "LSTMAE":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        xt, xv = self._to_tensor(X), self._to_tensor(X_val)
        self.net = _Net(self.n_feat, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=LSTM_LR)
        loss_fn = nn.MSELoss()

        best, best_state, bad = float("inf"), None, 0
        for _ in range(LSTM_MAX_EPOCHS):
            self.net.train()
            order = rng.permutation(len(xt))
            for i in range(0, len(order), LSTM_BATCH):
                batch = xt[order[i : i + LSTM_BATCH]]
                opt.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                opt.step()
            train_loss, val_loss = self._loss(xt), self._loss(xv)
            self.history.append((train_loss, val_loss))
            if val_loss < best - 1e-5:
                best, best_state, bad = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= LSTM_PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        self.best_epoch = int(np.argmin([v for _, v in self.history])) + 1
        return self

    @torch.no_grad()
    def _loss(self, x: torch.Tensor) -> float:
        self.net.eval()
        return float(((self.net(x) - x) ** 2).mean())

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        x = self._to_tensor(X)
        return ((self.net(x) - x) ** 2).mean(dim=(1, 2)).numpy()
