from src.split import split_units


def test_split_is_disjoint_and_complete():
    s = split_units(range(1, 101), val_fraction=0.2, seed=42)
    assert len(s["val"]) == 20 and len(s["fit"]) == 80
    assert not set(s["fit"]) & set(s["val"])
    assert set(s["fit"]) | set(s["val"]) == set(range(1, 101))


def test_split_is_deterministic():
    assert split_units(range(1, 101), seed=42) == split_units(range(1, 101), seed=42)
    assert split_units(range(1, 101), seed=42) != split_units(range(1, 101), seed=7)
