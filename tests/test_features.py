"""Testes das funções de preparação de dados (dados sintéticos, sem internet)."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FAILURE_MODE_COLUMNS,
    add_features,
    build_feature_row,
    clean_column_name,
    prepare_features,
    split_data,
)


def make_raw(n: int = 40) -> pd.DataFrame:
    """DataFrame sintético com as mesmas colunas do AI4I 2020."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"X{i}" for i in range(n)],
        "Type": rng.choice(["L", "M", "H"], n),
        "Air temperature [K]": rng.normal(300, 2, n).round(1),
        "Process temperature [K]": rng.normal(310, 1, n).round(1),
        "Rotational speed [rpm]": rng.integers(1200, 2000, n),
        "Torque [Nm]": rng.normal(40, 5, n).round(1),
        "Tool wear [min]": rng.integers(0, 250, n),
        "Machine failure": ([1, 0] * (n // 2)),
    })
    for col in FAILURE_MODE_COLUMNS:
        df[col] = 0
    return df


def test_clean_column_name():
    assert clean_column_name("Air temperature [K]") == "air_temperature_k"
    assert clean_column_name("Rotational speed [rpm]") == "rotational_speed_rpm"
    assert clean_column_name("Torque [Nm]") == "torque_nm"


def test_add_features_potencia_e_diferenca_termica():
    df = pd.DataFrame({
        "torque_nm": [10.0],
        "rotational_speed_rpm": [60.0],  # 60 rpm = 2π rad/s
        "process_temperature_k": [310.5],
        "air_temperature_k": [300.0],
    })
    out = add_features(df)
    assert out["power_w"].iloc[0] == pytest.approx(10.0 * 2 * np.pi, abs=0.01)
    assert out["temp_diff_k"].iloc[0] == pytest.approx(10.5)
    assert "power_w" not in df.columns  # não altera o original


def test_prepare_features_remove_vazamento_e_codifica_type():
    X, y = prepare_features(make_raw())
    assert "udi" not in X.columns and "product_id" not in X.columns
    for col in FAILURE_MODE_COLUMNS:
        assert col.lower() not in X.columns
    assert set(X["type"].unique()) <= {0, 1, 2}
    assert {"power_w", "temp_diff_k"} <= set(X.columns)
    assert len(X) == len(y) == 40
    assert set(y.unique()) == {0, 1}


def test_prepare_features_type_invalido():
    raw = make_raw()
    raw.loc[0, "Type"] = "Z"
    with pytest.raises(ValueError):
        prepare_features(raw)


def test_build_feature_row():
    row = build_feature_row("M", 300.0, 310.0, 1500, 40.0, 100)
    assert row.shape == (1, 8)
    assert row["type"].iloc[0] == 1
    assert row["temp_diff_k"].iloc[0] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        build_feature_row("X", 300.0, 310.0, 1500, 40.0, 100)


def test_split_data_estratificado():
    X, y = prepare_features(make_raw(100))
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    assert len(X_te) == 20 and len(X_tr) == 80
    assert y_te.mean() == pytest.approx(y.mean(), abs=0.05)
