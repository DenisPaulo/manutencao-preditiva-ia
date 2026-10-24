"""Testes da Versão 2 — Previsão de Vida Útil Remanescente (RUL) com NASA C-MAPSS (FD001)."""

from pathlib import Path

import numpy as np
import pytest

from src.cmapss import (
    FLAT_COLUMNS,
    INFORMATIVE_SENSORS,
    RUL_CAP,
    RUL_METRICS_PATH,
    RUL_MODEL_PATH,
    RUL_SAMPLE_PATH,
    WINDOW,
    classify_maintenance_zone,
    compute_piecewise_rul,
    compute_rul_metrics,
    explain_rul_row,
    feature_label,
    load_fd001,
    load_rul_metrics,
    load_rul_model,
    load_rul_sample_engines,
    make_rul_baseline,
    make_rul_xgb,
    make_synthetic_fd001,
    nasa_scoring_function,
    predict_rul,
    prepare_rul_features,
    prepare_test_last_cycle,
)


def test_make_synthetic_fd001_e_fallback_offline(tmp_path: Path):
    train_df, test_df, rul_true, origem = load_fd001(
        data_dir=tmp_path, download=False, allow_synthetic_fallback=True
    )
    assert "Sintético" in origem
    assert train_df["unit_id"].nunique() == 12
    assert test_df["unit_id"].nunique() == len(rul_true) == 8
    with pytest.raises(FileNotFoundError):
        load_fd001(data_dir=tmp_path, download=False, allow_synthetic_fallback=False)


def test_compute_piecewise_rul_aplica_teto_125():
    train_df, _, _ = make_synthetic_fd001(n_train=3, n_test=2, seed=0)
    y = compute_piecewise_rul(train_df, rul_cap=RUL_CAP)
    assert len(y) == len(train_df)
    assert y.max() == RUL_CAP
    assert y.min() == 0.0
    # Último ciclo de cada motor de treino deve ter RUL = 0
    last_idx = train_df.groupby("unit_id")["cycle"].idxmax()
    assert (y.loc[last_idx] == 0.0).all()


def test_prepare_rul_features_remove_constantes_e_cria_janela():
    train_df, _, _ = make_synthetic_fd001(n_train=2, n_test=2, seed=1)
    X = prepare_rul_features(train_df, window=WINDOW)
    # 1 coluna cycle + 14 sensores x 4 (bruto, média, desvio, delta) = 57 colunas
    assert X.shape == (len(train_df), 1 + 4 * len(INFORMATIVE_SENSORS))
    for flat in FLAT_COLUMNS:
        assert flat not in X.columns
    for s in INFORMATIVE_SENSORS:
        assert {s, f"{s}_mean{WINDOW}", f"{s}_std{WINDOW}", f"{s}_delta"} <= set(X.columns)
    # Nos primeiros ciclos do motor, o delta no ciclo 1 deve ser zero
    nth_rows = train_df.groupby("unit_id").nth(WINDOW - 1).index
    assert np.allclose(X.loc[nth_rows, "s11_delta"], 0.0, atol=1e-3)

    with pytest.raises(ValueError):
        prepare_rul_features(train_df.drop(columns=["s11"]))


def test_prepare_test_last_cycle_seleciona_1_linha_por_motor():
    _, test_df, rul_true = make_synthetic_fd001(n_train=3, n_test=5, seed=2)
    X_last, y_last = prepare_test_last_cycle(test_df, rul_true)
    assert len(X_last) == len(y_last) == 5
    assert (y_last <= RUL_CAP).all()


def test_nasa_scoring_function_assimetrica():
    y_true = np.array([50.0])
    # Erro atrasado (+20 ciclos: superestima a vida útil) deve ter penalidade maior que erro adiantado (-20 ciclos)
    score_atrasado = nasa_scoring_function(y_true, np.array([70.0]))
    score_adiantado = nasa_scoring_function(y_true, np.array([30.0]))
    assert nasa_scoring_function(y_true, y_true) == pytest.approx(0.0)
    assert score_atrasado > score_adiantado > 0.0

    m = compute_rul_metrics([20.0, 50.0, 100.0], [25.0, 45.0, 100.0])
    assert m["rmse_ciclos"] > 0 and m["mae_ciclos"] > 0
    assert 0.8 < m["r2"] <= 1.0
    assert m["previsoes_atrasadas"] == 1 and m["previsoes_adiantadas"] == 2


def test_classify_maintenance_zone():
    assert classify_maintenance_zone(8.0) == "Crítico"
    assert classify_maintenance_zone(20.0) == "Crítico"
    assert classify_maintenance_zone(20.1) == "Atenção"
    assert classify_maintenance_zone(50.0) == "Atenção"
    assert classify_maintenance_zone(50.1) == "Saudável"


def test_treino_baseline_e_xgb_em_dados_sinteticos():
    train_df, test_df, rul_true = make_synthetic_fd001(n_train=6, n_test=4, seed=7)
    X_tr = prepare_rul_features(train_df)
    y_tr = compute_piecewise_rul(train_df)
    X_te, y_te = prepare_test_last_cycle(test_df, rul_true)

    base = make_rul_baseline().fit(X_tr, y_tr)
    pred_b = predict_rul(base, X_te)
    assert pred_b.shape == (4,)
    assert ((pred_b >= 0) & (pred_b <= RUL_CAP)).all()

    xgb_m = make_rul_xgb(n_estimators=15, max_depth=3).fit(X_tr, y_tr)
    pred_x = predict_rul(xgb_m, X_te)
    assert ((pred_x >= 0) & (pred_x <= RUL_CAP)).all()
    assert compute_rul_metrics(y_te, pred_x)["rmse_ciclos"] >= 0


def test_artefatos_v2_versionados_e_inferencia():
    assert RUL_MODEL_PATH.exists() and RUL_METRICS_PATH.exists() and RUL_SAMPLE_PATH.exists()
    metrics = load_rul_metrics()
    samples = load_rul_sample_engines()
    model = load_rul_model()

    assert metrics["configuracao"]["rul_cap"] == RUL_CAP
    assert len(metrics["features"]) == 57
    assert metrics["teste"]["xgboost_regressor"]["rmse_ciclos"] < metrics["teste"]["baseline_ridge"]["rmse_ciclos"]
    assert metrics["teste"]["xgboost_regressor"]["nasa_score"] < metrics["teste"]["baseline_ridge"]["nasa_score"]

    # Motor crítico (#31) deve ter RUL prevista muito menor que motor saudável (#10)
    crit = samples["motores_destaque"]["31"]
    saud = samples["motores_destaque"]["10"]
    res_crit = explain_rul_row(crit["ultimo_ciclo_features"])
    res_saud = explain_rul_row(saud["ultimo_ciclo_features"])
    assert res_crit["zona"] == "Crítico"
    assert res_saud["zona"] == "Saudável"
    assert res_crit["rul_previsto"] < 20.0 < 50.0 < res_saud["rul_previsto"]
    assert len(res_crit["shap"]) == 57
    assert "Pressão estática HPC" in feature_label("s11_delta")
    assert len(predict_rul(model, [list(crit["ultimo_ciclo_features"].values())])) == 1
