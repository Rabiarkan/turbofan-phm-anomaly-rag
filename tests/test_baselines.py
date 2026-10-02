import numpy as np

from src.baselines import BASELINES, SPE, T2


def _healthy(n=500, d=6, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 1))
    return latent @ np.ones((1, d)) + 0.3 * rng.normal(size=(n, d))  # one shared direction + noise


def test_baselines_flag_a_large_shift_along_the_healthy_direction():
    # SPE is excluded on purpose: a shift along the healthy correlation direction
    # stays inside the PCA subspace, so SPE should NOT see it (tested below).
    X = _healthy()
    far = np.full((1, X.shape[1]), 6.0)
    for name, cls in BASELINES.items():
        if name.startswith("pca_spe"):
            continue
        m = cls().fit(X)
        assert m.score(far)[0] > np.quantile(m.score(X), 0.99), name


def test_spe_and_t2_separate_two_kinds_of_shift():
    X = _healthy()
    d = X.shape[1]
    along = np.full((1, d), 4.0)  # moves along the healthy correlation direction
    across = np.array([[3.0, -3.0] + [0.0] * (d - 2)])  # breaks the correlation structure
    spe, t2 = SPE(n_components=0.8).fit(X), T2(n_components=0.8).fit(X)
    assert t2.score(along)[0] > np.quantile(t2.score(X), 0.99)
    assert spe.score(along)[0] < np.quantile(spe.score(X), 0.99)
    assert spe.score(across)[0] > np.quantile(spe.score(X), 0.99)
