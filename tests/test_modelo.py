"""Testes de treino/avaliação/explicação com dados sintéticos e o modelo versionado."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import bootstrap_ci, compute_metrics, naive_metrics
from src.explain import explain_row
from src.features import build_feature_row
from src.train import (
    THRESHOLD_PATH,
    best_f_beta_threshold,
    load_threshold_config,
    load_xgb_model,
    make_baseline,
    make_xgb,
)

Y = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
PROBA = np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.8, 0.2, 0.4, 0.1, 0.05])


def test_compute_metrics():
    m = compute_metrics(Y, PROBA, 0.5)
    # pred>=0.5: positivos nos índices 3, 4, 5 -> VP=2, FP=1, FN=1, VN=6
    assert m["matriz_confusao"] == {"vn": 6, "fp": 1, "fn": 1, "vp": 2}
    assert m["recall"] == pytest.approx(0.6667, abs=1e-4)
    assert m["precisao"] == pytest.approx(0.6667, abs=1e-4)
    assert m["falhas_detectadas"] == "2 de 3"
    assert m["alarmes_falsos"] == 1


def test_naive_metrics():
    m = naive_metrics(Y)
    assert m["recall"] == 0.0
    assert m["acuracia"] == pytest.approx(0.7)
    assert m["falhas_detectadas"] == "0 de 3"


def test_bootstrap_ci_reprodutivel():
    a = bootstrap_ci(Y, PROBA, 0.5, n=50, seed=1)
    b = bootstrap_ci(Y, PROBA, 0.5, n=50, seed=1)
    assert a == b
    for k in ("recall", "precisao", "pr_auc"):
        lo, hi = a[k]
        assert 0 <= lo <= hi <= 1


def test_best_f_beta_threshold():
    r = best_f_beta_threshold(Y, PROBA)
    assert 0 < r["threshold"] <= 1
    assert 0 <= r["f2_oof"] <= 1


def test_baseline_e_xgb_treinam_em_dados_sinteticos():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = (X["a"] + 0.1 * rng.normal(size=120) > 1).astype(int)
    assert y.sum() > 0
    base = make_baseline().fit(X, y)
    assert base.predict_proba(X).shape == (120, 2)
    model = make_xgb(scale_pos_weight=5.0, n_estimators=10, max_depth=2).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_config_do_limiar_versionada():
    config = load_threshold_config()
    assert 0 < config["threshold"] < 1
    assert config["type_mapping"] == {"L": 0, "M": 1, "H": 2}
    assert json.loads(THRESHOLD_PATH.read_text(encoding="utf-8")) == config


def test_modelo_versionado_preve_probabilidade():
    config = load_threshold_config()
    model = load_xgb_model()
    # Ferramenta muito gasta + torque alto: caso de risco; operação folgada: caso normal
    risco = build_feature_row("L", 303.0, 312.0, 1300, 65.0, 240)
    normal = build_feature_row("M", 298.0, 308.5, 1500, 40.0, 20)
    p_risco = float(model.predict_proba(risco[config["features"]])[:, 1][0])
    p_normal = float(model.predict_proba(normal[config["features"]])[:, 1][0])
    assert 0 <= p_normal <= 1 and 0 <= p_risco <= 1
    assert p_risco > p_normal


def test_explain_row():
    row = build_feature_row("M", 298.0, 308.5, 1500, 40.0, 20)
    res = explain_row(row)
    assert 0 <= res["probabilidade"] <= 1
    assert isinstance(res["alarme"], bool)
    assert len(res["shap"]) == 8
