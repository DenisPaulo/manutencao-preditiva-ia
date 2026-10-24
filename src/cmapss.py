"""Versão 2 — Previsão de Vida Útil Remanescente (RUL) com NASA C-MAPSS (FD001).

Uso pela linha de comando (a partir da raiz do projeto):

    python -m src.cmapss

Fluxo completo (dados, preparação, treino, avaliação e explicação SHAP):
1. Lê os arquivos brutos ``train_FD001.txt``, ``test_FD001.txt`` e ``RUL_FD001.txt`` em
   ``data/raw/cmapss/`` (baixando de espelho público se necessário, ou gerando uma amostra
   sintética determinística em ambientes sem internet).
2. Remove configurações e sensores planos/constantes no FD001 (``op_setting_3`` e sensores
   ``s1, s5, s6, s10, s16, s18, s19``) e mantém os 14 sensores informativos.
3. Aplica o teto linear por partes (*piecewise linear RUL*) em ``RUL_CAP = 125`` ciclos.
4. Cria variáveis de trajetória por motor (janela móvel de 5 ciclos: média, desvio-padrão e
   variação em relação aos ciclos iniciais do próprio motor).
5. Treina baseline (``Ridge``) e modelo principal (``XGBRegressor``) nos 100 motores de
   treino e avalia no último ciclo observado dos 100 motores de teste oficiais.
6. Calcula RMSE, MAE, R² e a função de pontuação assimétrica da NASA (PHM08), salva
   ``models/xgb_rul_fd001.json``, ``models/rul_metrics.json`` e ``models/rul_sample_engines.json``
   e gera as figuras em ``reports/figures/``.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import PROJECT_ROOT

CMAPSS_DIR = PROJECT_ROOT / "data" / "raw" / "cmapss"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RUL_MODEL_PATH = MODELS_DIR / "xgb_rul_fd001.json"
RUL_METRICS_PATH = MODELS_DIR / "rul_metrics.json"
RUL_SAMPLE_PATH = MODELS_DIR / "rul_sample_engines.json"

RANDOM_STATE = 42
RUL_CAP = 125
WINDOW = 5

# Zonas de decisão para manutenção preventiva por condição (em ciclos restantes)
CRITICAL_ZONE = 20
ATTENTION_ZONE = 50

RAW_COLUMNS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensores e ajustes sem variação útil no subconjunto FD001 (nível do mar, falha no HPC)
FLAT_COLUMNS = ["op_setting_3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# 14 sensores que variam ao longo da degradação da turbina no FD001
INFORMATIVE_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]

SENSOR_SHORT_LABELS = {
    "s2": "s2 Temp. saída LPC [°R]",
    "s3": "s3 Temp. saída HPC [°R]",
    "s4": "s4 Temp. saída LPT [°R]",
    "s7": "s7 Pressão saída HPC [psia]",
    "s8": "s8 Rotação física fan [rpm]",
    "s9": "s9 Rotação física núcleo [rpm]",
    "s11": "s11 Pressão estática HPC [psia]",
    "s12": "s12 Razão fluxo combustível/Ps30",
    "s13": "s13 Rotação corrigida fan [rpm]",
    "s14": "s14 Rotação corrigida núcleo [rpm]",
    "s15": "s15 Razão de bypass (BPR)",
    "s17": "s17 Entalpia de sangria",
    "s20": "s20 Fluxo resfriamento HPT",
    "s21": "s21 Fluxo resfriamento LPT",
}

MIRROR_URLS = [
    "https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData",
    "https://raw.githubusercontent.com/ericlrf/rul/main/CMAPSSData",
]

EXPECTED_COUNTS = {
    "train_FD001.txt": 20_631,
    "test_FD001.txt": 13_096,
    "RUL_FD001.txt": 100,
}

XGB_RUL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
}


def feature_label(col: str) -> str:
    """Rótulo amigável em português para cada variável criada a partir dos sensores."""
    if col == "cycle":
        return "Ciclo atual do motor"
    for s, nome in SENSOR_SHORT_LABELS.items():
        if col == s:
            return f"{nome} (leitura atual)"
        if col == f"{s}_mean{WINDOW}":
            return f"{nome} (média {WINDOW} ciclos)"
        if col == f"{s}_std{WINDOW}":
            return f"{nome} (desvio {WINDOW} ciclos)"
        if col == f"{s}_delta":
            return f"{nome} (Δ vs início)"
    return col


def download_cmapss(force: bool = False, timeout: int = 60) -> Path:
    """Baixa os 3 arquivos do FD001 para ``data/raw/cmapss/`` se ainda não existirem."""
    CMAPSS_DIR.mkdir(parents=True, exist_ok=True)
    for filename in EXPECTED_COUNTS:
        dest = CMAPSS_DIR / filename
        if dest.exists() and not force:
            continue
        last_err: Exception | None = None
        for base_url in MIRROR_URLS:
            url = f"{base_url}/{filename}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "manutencao-preditiva-ia"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = resp.read()
                tmp = dest.with_suffix(".txt.tmp")
                tmp.write_bytes(payload)
                tmp.replace(dest)
                last_err = None
                break
            except Exception as err:  # pragma: no cover - tenta próximo espelho
                last_err = err
        if last_err is not None:
            raise RuntimeError(f"Não foi possível baixar {filename}: {last_err}") from last_err
    return CMAPSS_DIR


def make_synthetic_fd001(
    n_train: int = 12,
    n_test: int = 8,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Gera um conjunto FD001 sintético e determinístico para testes e uso offline."""
    rng = np.random.default_rng(seed)

    # Sinais crescentes (+1) ou decrescentes (-1) com a degradação, imitando o HPC real
    directions = {
        "s2": 1.0, "s3": 1.0, "s4": 1.0, "s7": -1.0, "s8": 1.0, "s9": 1.0, "s11": 1.0,
        "s12": -1.0, "s13": 1.0, "s14": 1.0, "s15": 1.0, "s17": 1.0, "s20": -1.0, "s21": -1.0,
    }
    baselines = {
        "s2": 642.5, "s3": 1588.0, "s4": 1405.0, "s7": 553.5, "s8": 2388.1, "s9": 9055.0,
        "s11": 47.4, "s12": 521.5, "s13": 2388.1, "s14": 8135.0, "s15": 8.42, "s17": 392.0,
        "s20": 38.9, "s21": 23.3,
    }

    def _sim_engine(unit_id: int, total_cycles: int, cutoff: int) -> pd.DataFrame:
        cycles = np.arange(1, cutoff + 1)
        rem = total_cycles - cycles
        # Degradação quadrática suave nos últimos 125 ciclos
        deg = np.clip((RUL_CAP - rem) / RUL_CAP, 0.0, 1.0) ** 1.6
        data: dict[str, np.ndarray] = {
            "unit_id": np.full(cutoff, unit_id, dtype=int),
            "cycle": cycles,
            "op_setting_1": rng.normal(0.0, 0.002, cutoff).round(4),
            "op_setting_2": rng.normal(0.0, 0.0003, cutoff).round(4),
            "op_setting_3": np.full(cutoff, 100.0),
        }
        for i in range(1, 22):
            s = f"s{i}"
            if s in directions:
                noise = rng.normal(0.0, 0.12, cutoff)
                data[s] = (baselines[s] + directions[s] * (2.5 * deg + noise)).round(4)
            else:
                data[s] = np.full(cutoff, float(i * 10))
        return pd.DataFrame(data)[RAW_COLUMNS]

    train_parts = []
    for unit_id in range(1, n_train + 1):
        life = int(rng.integers(150, 260))
        train_parts.append(_sim_engine(unit_id, life, life))

    test_parts = []
    rul_list = []
    for unit_id in range(1, n_test + 1):
        life = int(rng.integers(160, 260))
        true_rul = int(rng.integers(8, 135))
        cutoff = max(25, life - true_rul)
        test_parts.append(_sim_engine(unit_id, cutoff + true_rul, cutoff))
        rul_list.append(true_rul)

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
        pd.Series(rul_list, name="rul", dtype="int64"),
    )


def load_fd001(
    data_dir: Path = CMAPSS_DIR,
    download: bool = False,
    allow_synthetic_fallback: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str]:
    """Carrega ``(train_df, test_df, rul_true, origem)`` do FD001."""
    files_exist = all((data_dir / f).exists() for f in EXPECTED_COUNTS)
    if not files_exist and download:
        try:
            download_cmapss()
            files_exist = all((data_dir / f).exists() for f in EXPECTED_COUNTS)
        except Exception:
            if not allow_synthetic_fallback:
                raise

    if files_exist:
        train_df = pd.read_csv(data_dir / "train_FD001.txt", sep=r"\s+", header=None, names=RAW_COLUMNS)
        test_df = pd.read_csv(data_dir / "test_FD001.txt", sep=r"\s+", header=None, names=RAW_COLUMNS)
        rul_true = pd.read_csv(data_dir / "RUL_FD001.txt", sep=r"\s+", header=None, names=["rul"])["rul"]
        return train_df, test_df, rul_true.astype("int64"), "NASA C-MAPSS FD001 (real)"

    if allow_synthetic_fallback:
        train_df, test_df, rul_true = make_synthetic_fd001()
        return train_df, test_df, rul_true, "Sintético determinístico (fallback)"

    raise FileNotFoundError(
        f"Arquivos FD001 não encontrados em {data_dir}. Rode `python -m src.cmapss` para baixar."
    )


def compute_piecewise_rul(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.Series:
    """Calcula a RUL de treino por motor com teto linear por partes em ``rul_cap`` ciclos.

    Em turbinas saudáveis no início da operação, os sensores permanecem estáveis;
    a degradação visível começa cerca de 125 ciclos antes da falha. Limitar o alvo
    em ``rul_cap = 125`` evita penalizar o modelo por não distinguir 180 de 250 ciclos.
    """
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    rul = (max_cycle - df["cycle"]).clip(lower=0, upper=rul_cap)
    return rul.astype(float).rename("rul")


def prepare_rul_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Remove sensores constantes e adiciona estatísticas de janela móvel por motor.

    Para cada um dos 14 sensores informativos do FD001 são mantidos/criados:
    - valor bruto no ciclo atual (``s2`` ... ``s21``);
    - média móvel nos últimos ``window`` ciclos do motor (``_mean5``);
    - desvio-padrão móvel nos últimos ``window`` ciclos (``_std5``);
    - diferença entre a média móvel atual e a média inicial do motor (``_delta``),
      capturando o quanto o motor se afastou da sua própria condição saudável inicial.
    """
    missing = set(["unit_id", "cycle"] + INFORMATIVE_SENSORS) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no DataFrame C-MAPSS: {sorted(missing)}")

    ordered = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    grouped = ordered.groupby("unit_id", sort=False)

    features: dict[str, pd.Series] = {"cycle": ordered["cycle"].astype(float)}
    for s in INFORMATIVE_SENSORS:
        col = ordered[s].astype(float)
        rmean = grouped[s].transform(lambda x: x.rolling(window, min_periods=1).mean())
        rstd = grouped[s].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0.0))
        init_mean = grouped[s].transform(lambda x: x.iloc[:window].mean())
        features[s] = col
        features[f"{s}_mean{window}"] = rmean.round(4)
        features[f"{s}_std{window}"] = rstd.round(4)
        features[f"{s}_delta"] = (rmean - init_mean).round(4)

    X = pd.DataFrame(features)
    leaked_flat = set(FLAT_COLUMNS) & set(X.columns)
    assert not leaked_flat, f"Sensores planos não deveriam estar em X: {leaked_flat}"
    return X


def prepare_test_last_cycle(
    test_df: pd.DataFrame,
    rul_true: pd.Series,
    rul_cap: int = RUL_CAP,
    window: int = WINDOW,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepara as features de toda a série de teste e seleciona o último ciclo de cada motor."""
    ordered = test_df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    X_all = prepare_rul_features(ordered, window=window)
    last_idx = ordered.groupby("unit_id")["cycle"].idxmax().sort_index()
    X_last = X_all.loc[last_idx].reset_index(drop=True)
    y_last = rul_true.reset_index(drop=True).clip(lower=0, upper=rul_cap).astype(float).rename("rul")
    return X_last, y_last


def nasa_scoring_function(y_true, y_pred) -> float:
    """Função de pontuação assimétrica da NASA (PHM08).

    Penaliza previsões atrasadas/otimistas (``d = y_pred - y_true > 0``, divisor 10)
    mais severamente do que previsões adiantadas/conservadoras (``d < 0``, divisor 13).
    Na manutenção industrial, prever mais vida útil do que a máquina realmente tem
    leva à quebra inesperada em operação; prever um pouco menos só antecipa a preventiva.
    Quanto menor o valor, melhor (0 = previsão exata).
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalties))


def compute_rul_metrics(y_true, y_pred) -> dict:
    """Calcula RMSE, MAE, R² e pontuação assimétrica da NASA no conjunto de teste."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    score = nasa_scoring_function(y_true, y_pred)
    d = y_pred - y_true
    return {
        "rmse_ciclos": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "mae_ciclos": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "nasa_score": round(score, 1),
        "nasa_score_por_motor": round(score / len(y_true), 2),
        "previsoes_atrasadas": int(np.sum(d > 0)),
        "previsoes_adiantadas": int(np.sum(d <= 0)),
    }


def classify_maintenance_zone(rul_cycles: float) -> str:
    """Classifica a vida útil remanescente em zona de decisão para manutenção."""
    if rul_cycles <= CRITICAL_ZONE:
        return "Crítico"
    if rul_cycles <= ATTENTION_ZONE:
        return "Atenção"
    return "Saudável"


def make_rul_baseline() -> "sklearn.pipeline.Pipeline":
    """Modelo base linear (padronização + regressão Ridge)."""
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def make_rul_xgb(**params) -> xgb.XGBRegressor:
    """Modelo principal XGBoost para regressão de RUL."""
    cfg = {**XGB_RUL_PARAMS, **params}
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **cfg,
    )


def predict_rul(model, X: pd.DataFrame, rul_cap: int = RUL_CAP) -> np.ndarray:
    """Prevê a RUL em ciclos limitada ao intervalo válido ``[0, rul_cap]``."""
    return np.clip(np.asarray(model.predict(X), dtype=float), 0.0, float(rul_cap))


@lru_cache(maxsize=1)
def load_rul_model(path: Path = RUL_MODEL_PATH) -> xgb.XGBRegressor:
    """Carrega o modelo XGBRegressor salvo em ``models/xgb_rul_fd001.json``."""
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


@lru_cache(maxsize=1)
def get_rul_explainer() -> shap.TreeExplainer:
    """Cria e mantém em cache o TreeExplainer do modelo de RUL."""
    return shap.TreeExplainer(load_rul_model())


def load_rul_metrics(path: Path = RUL_METRICS_PATH) -> dict:
    """Lê as métricas e metadados do modelo V2."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rul_sample_engines(path: Path = RUL_SAMPLE_PATH) -> dict:
    """Lê as trajetórias de motores reais salvas para uso imediato no painel."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def explain_rul_row(row: pd.DataFrame | pd.Series | dict) -> dict:
    """Explica a previsão de RUL (em ciclos) de uma única leitura/ciclo com SHAP."""
    metrics = load_rul_metrics()
    features = metrics["features"]
    if isinstance(row, dict):
        row = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        row = row.to_frame().T
    row = row[features].astype(float)
    if len(row) != 1:
        raise ValueError(f"explain_rul_row espera 1 linha, recebeu {len(row)}.")

    explainer = get_rul_explainer()
    expl = explainer(row)[0]
    expl.feature_names = [feature_label(f) for f in features]
    rul_pred = float(predict_rul(load_rul_model(), row)[0])
    valores = pd.Series(expl.values, index=features)
    return {
        "rul_previsto": round(rul_pred, 1),
        "zona": classify_maintenance_zone(rul_pred),
        "valor_base": float(expl.base_values),
        "shap": valores.reindex(valores.abs().sort_values(ascending=False).index),
        "explanation": expl,
    }


def _build_sample_engines_payload(
    test_df: pd.DataFrame,
    rul_true: pd.Series,
    model: xgb.XGBRegressor,
    features: list[str],
) -> dict:
    """Salva trajetórias completas de motores representativos + resumo dos 100 motores."""
    ordered = test_df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    X_all = prepare_rul_features(ordered)
    preds_all = predict_rul(model, X_all[features])
    ordered["rul_pred"] = np.round(preds_all, 1)

    last_rows = ordered.groupby("unit_id").tail(1).reset_index(drop=True)
    summary_100 = []
    for i, r in last_rows.iterrows():
        t_rul = float(min(rul_true.iloc[i], RUL_CAP))
        p_rul = float(r["rul_pred"])
        summary_100.append({
            "unit_id": int(r["unit_id"]),
            "ciclos_observados": int(r["cycle"]),
            "rul_real": round(t_rul, 1),
            "rul_previsto": round(p_rul, 1),
            "erro_ciclos": round(p_rul - t_rul, 1),
            "zona": classify_maintenance_zone(p_rul),
        })

    # Escolhe motores reais das 3 zonas (Crítico, Atenção, Saudável) para inspeção ciclo a ciclo
    by_zone = {
        "🚨 Motor #31 — Zona crítica (RUL real: 8 ciclos)": 31,
        "🚨 Motor #34 — Zona crítica (RUL real: 7 ciclos)": 34,
        "⚠️ Motor #18 — Zona de atenção (RUL real: 28 ciclos)": 18,
        "⚠️ Motor #32 — Zona de atenção (RUL real: 48 ciclos)": 32,
        "✅ Motor #10 — Zona saudável (RUL real: 96 ciclos)": 10,
        "✅ Motor #4 — Zona saudável (RUL real: 82 ciclos)": 4,
    }
    available_ids = sorted(ordered["unit_id"].unique().tolist())
    engines = {}
    for label, uid in by_zone.items():
        if uid not in available_ids:
            uid = available_ids[len(engines) % len(available_ids)]
        sub_df = ordered[ordered["unit_id"] == uid].reset_index(drop=True)
        sub_X = X_all.loc[ordered["unit_id"] == uid, features].reset_index(drop=True)
        idx_unit = available_ids.index(uid)
        true_final_rul = float(min(rul_true.iloc[idx_unit], RUL_CAP))
        max_c = int(sub_df["cycle"].max())
        # RUL real em cada ciclo observado (limitada em RUL_CAP)
        rul_real_curve = np.minimum(true_final_rul + (max_c - sub_df["cycle"].to_numpy()), RUL_CAP)

        engines[str(uid)] = {
            "unit_id": int(uid),
            "rotulo": label,
            "ciclos_observados": max_c,
            "rul_real_final": round(true_final_rul, 1),
            "rul_previsto_final": round(float(sub_df["rul_pred"].iloc[-1]), 1),
            "zona_final": classify_maintenance_zone(float(sub_df["rul_pred"].iloc[-1])),
            "trajetoria": {
                "cycle": sub_df["cycle"].astype(int).tolist(),
                "rul_real": np.round(rul_real_curve, 1).tolist(),
                "rul_pred": sub_df["rul_pred"].round(1).tolist(),
                "s2": sub_df["s2"].round(2).tolist(),
                "s4": sub_df["s4"].round(2).tolist(),
                "s7": sub_df["s7"].round(2).tolist(),
                "s11": sub_df["s11"].round(2).tolist(),
                "s12": sub_df["s12"].round(2).tolist(),
            },
            "ultimo_ciclo_features": {k: round(float(sub_X.iloc[-1][k]), 4) for k in features},
        }

    return {
        "rul_cap": RUL_CAP,
        "window": WINDOW,
        "zonas": {"critico_max": CRITICAL_ZONE, "atencao_max": ATTENTION_ZONE},
        "motores_destaque": engines,
        "resumo_teste": summary_100,
    }


def plot_rul_predictions(y_true: np.ndarray, pred_base: np.ndarray, pred_xgb: np.ndarray):
    """Figura 1: RUL previsto vs RUL real nos motores do conjunto de teste."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Painel esquerdo: dispersão previsto vs real
    ax = axes[0]
    ax.axhspan(0, CRITICAL_ZONE, color="#FDECEA", alpha=0.7, label="Zona crítica (≤ 20 ciclos)")
    ax.axhspan(CRITICAL_ZONE, ATTENTION_ZONE, color="#FFF8E1", alpha=0.7, label="Zona de atenção (21–50)")
    ax.plot([0, RUL_CAP], [0, RUL_CAP], "k--", lw=1.2, label="Previsão ideal (y = x)")
    ax.scatter(y_true, pred_base, color="#8C8C8C", alpha=0.55, s=36, label="Baseline (Ridge)")
    ax.scatter(y_true, pred_xgb, color="#C44E52", alpha=0.85, s=46, edgecolor="black", lw=0.4, label="XGBRegressor")
    ax.set_xlabel("RUL real no último ciclo [ciclos]")
    ax.set_ylabel("RUL previsto [ciclos]")
    ax.set_title("RUL previsto vs. real (100 motores de teste)")
    ax.set_xlim(0, RUL_CAP + 5)
    ax.set_ylim(0, RUL_CAP + 5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)

    # Painel direito: motores ordenados pela RUL real
    ax2 = axes[1]
    order = np.argsort(y_true)
    idx = np.arange(1, len(y_true) + 1)
    ax2.plot(idx, y_true[order], "k-", lw=2, label="RUL real (com teto 125)")
    ax2.plot(idx, pred_xgb[order], color="#C44E52", lw=1.6, marker="o", ms=3.5, alpha=0.85, label="XGBRegressor")
    ax2.axhline(CRITICAL_ZONE, color="#B3261E", ls=":", lw=1, label="Limite crítico (20 ciclos)")
    ax2.set_xlabel("Motores do teste (ordenados pela RUL real)")
    ax2.set_ylabel("Vida útil remanescente [ciclos]")
    ax2.set_title("Acompanhamento por motor no teste FD001")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    return fig


def plot_degradation_and_nasa_curve(sample_payload: dict):
    """Figura 2: curva de degradação ciclo a ciclo de um motor até a zona crítica + função NASA."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Painel esquerdo: trajetória do motor #31 (ou primeiro disponível)
    engines = sample_payload["motores_destaque"]
    eng = engines.get("31") or next(iter(engines.values()))
    tr = eng["trajetoria"]
    ax = axes[0]
    ax.axhspan(0, CRITICAL_ZONE, color="#FDECEA", alpha=0.75, label="Crítico (≤ 20 ciclos)")
    ax.axhspan(CRITICAL_ZONE, ATTENTION_ZONE, color="#FFF8E1", alpha=0.75, label="Atenção (21–50 ciclos)")
    ax.plot(tr["cycle"], tr["rul_real"], "k--", lw=1.8, label="RUL real (teto 125)")
    ax.plot(tr["cycle"], tr["rul_pred"], color="#C44E52", lw=2.2, label="RUL previsto (XGBoost)")
    ax.set_xlabel("Ciclo de operação do motor")
    ax.set_ylabel("RUL [ciclos]")
    ax.set_title(f"Curva de degradação — Motor #{eng['unit_id']} (teste FD001)")
    ax.set_ylim(0, RUL_CAP + 8)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5)

    # Painel direito: assimetria da função NASA
    ax2 = axes[1]
    d = np.linspace(-50, 50, 201)
    s_nasa = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    ax2.plot(d, s_nasa, color="#1F4E78", lw=2.2, label="Penalidade NASA (PHM08)")
    ax2.axvline(0, color="black", ls="--", lw=1)
    ax2.fill_between(d[d >= 0], 0, s_nasa[d >= 0], color="#C44E52", alpha=0.18,
                     label="Erro atrasado (d > 0): superestima a vida → risco de quebra")
    ax2.fill_between(d[d <= 0], 0, s_nasa[d <= 0], color="#4C72B0", alpha=0.18,
                     label="Erro adiantado (d < 0): antecipa a manutenção")
    ax2.set_xlabel("Erro de previsão d = RUL previsto − RUL real [ciclos]")
    ax2.set_ylabel("Penalidade por motor")
    ax2.set_title("Por que a métrica NASA é assimétrica?")
    ax2.set_ylim(0, 80)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.2)

    fig.tight_layout()
    return fig


def plot_rul_shap_importance(model: xgb.XGBRegressor, X_test: pd.DataFrame):
    """Figura 3: importância global SHAP nas previsões de RUL do conjunto de teste."""
    explainer = shap.TreeExplainer(model)
    expl = explainer(X_test)
    expl.feature_names = [feature_label(f) for f in X_test.columns]
    plt.figure()
    shap.plots.bar(expl, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9.2, 5.2)
    plt.title("Importância global SHAP — Previsão de RUL (NASA C-MAPSS FD001)")
    plt.xlabel("Média de |valor SHAP| [ciclos de vida útil]")
    plt.tight_layout()
    return fig


def main() -> None:
    train_df, test_df, rul_true, origem = load_fd001(download=True, allow_synthetic_fallback=True)
    print(f"[cmapss] Origem dos dados: {origem}")
    print(
        f"[cmapss] Treino: {len(train_df)} linhas ({train_df['unit_id'].nunique()} motores) | "
        f"Teste: {len(test_df)} linhas ({test_df['unit_id'].nunique()} motores)"
    )

    X_train = prepare_rul_features(train_df, window=WINDOW)
    y_train = compute_piecewise_rul(train_df, rul_cap=RUL_CAP)
    X_test, y_test = prepare_test_last_cycle(test_df, rul_true, rul_cap=RUL_CAP, window=WINDOW)
    features = list(X_train.columns)
    print(f"[cmapss] Variáveis de entrada ({len(features)}): 14 sensores + média/desvio/delta ({WINDOW} ciclos)")

    # 1) Baseline (Ridge) e 2) Modelo principal (XGBRegressor)
    baseline = make_rul_baseline().fit(X_train, y_train)
    pred_base = predict_rul(baseline, X_test)
    m_base = compute_rul_metrics(y_test, pred_base)

    xgb_model = make_rul_xgb().fit(X_train, y_train)
    pred_xgb = predict_rul(xgb_model, X_test)
    m_xgb = compute_rul_metrics(y_test, pred_xgb)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(RUL_MODEL_PATH)
    load_rul_model.cache_clear()
    get_rul_explainer.cache_clear()

    # Importância SHAP global para registrar as principais variáveis no JSON
    expl = shap.TreeExplainer(xgb_model)(X_test)
    mean_shap = pd.Series(np.abs(expl.values).mean(axis=0), index=features).sort_values(ascending=False)
    top_shap = [
        {"feature": f, "rotulo": feature_label(f), "shap_medio_ciclos": round(float(v), 2)}
        for f, v in mean_shap.head(10).items()
    ]

    metrics_payload = {
        "gerado_em_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": origem,
        "versoes": {"xgboost": xgb.__version__, "scikit-learn": sklearn.__version__},
        "configuracao": {
            "rul_cap": RUL_CAP,
            "window": WINDOW,
            "sensores_removidos_constantes": FLAT_COLUMNS,
            "sensores_informativos": INFORMATIVE_SENSORS,
            "zonas_manutencao": {
                "critico": f"<= {CRITICAL_ZONE} ciclos (intervenção imediata)",
                "atencao": f"{CRITICAL_ZONE + 1} a {ATTENTION_ZONE} ciclos (programar preventiva)",
                "saudavel": f"> {ATTENTION_ZONE} ciclos (operação normal)",
            },
        },
        "treino": {
            "linhas": int(len(train_df)),
            "motores": int(train_df["unit_id"].nunique()),
            "num_features": len(features),
        },
        "teste": {
            "linhas_totais": int(len(test_df)),
            "motores_avaliados": int(len(y_test)),
            "baseline_ridge": m_base,
            "xgboost_regressor": m_xgb,
        },
        "hiperparametros_xgboost": XGB_RUL_PARAMS,
        "top_features_shap": top_shap,
        "features": features,
    }
    RUL_METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sample_payload = _build_sample_engines_payload(test_df, rul_true, xgb_model, features)
    RUL_SAMPLE_PATH.write_text(json.dumps(sample_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figs = {
        "rul_previsto_vs_real.png": plot_rul_predictions(y_test.to_numpy(), pred_base, pred_xgb),
        "rul_degradacao_motor.png": plot_degradation_and_nasa_curve(sample_payload),
        "rul_shap_importancia.png": plot_rul_shap_importance(xgb_model, X_test),
    }
    for nome, fig in figs.items():
        fig.savefig(FIGURES_DIR / nome, dpi=110, bbox_inches="tight")
        plt.close(fig)

    print(f"[cmapss] Baseline Ridge -> RMSE: {m_base['rmse_ciclos']} | MAE: {m_base['mae_ciclos']} | R²: {m_base['r2']} | NASA: {m_base['nasa_score']}")
    print(f"[cmapss] XGBRegressor   -> RMSE: {m_xgb['rmse_ciclos']} | MAE: {m_xgb['mae_ciclos']} | R²: {m_xgb['r2']} | NASA: {m_xgb['nasa_score']}")
    print(f"[cmapss] Artefatos salvos em {RUL_MODEL_PATH.relative_to(PROJECT_ROOT)}, {RUL_METRICS_PATH.relative_to(PROJECT_ROOT)} e {RUL_SAMPLE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
