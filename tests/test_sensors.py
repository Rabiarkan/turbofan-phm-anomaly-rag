from src.config import FEATURE_SENSORS
from src.sensors import SENSORS, phrase, query_text


def test_every_feature_sensor_has_a_description_and_unit():
    assert set(SENSORS) == set(FEATURE_SENSORS)
    assert all(s.description and s.unit for s in SENSORS.values())


def test_terms_only_add_words_missing_from_the_description():
    for name, s in SENSORS.items():
        for term in s.terms:
            assert term.lower() not in s.description.lower(), (name, term)


def test_phrase_shows_name_description_direction_and_size():
    assert (
        phrase("T50", 3.44) == "T50 (total temperature at LPT (low-pressure turbine) outlet) ↑ 3.4σ"
    )
    assert phrase("P30", -2.9).endswith("↓ 2.9σ")


def test_query_text_adds_handbook_terms():
    assert query_text("P30") == "total pressure at HPC outlet compressor discharge pressure"
    assert query_text("BPR") == "bypass ratio"
