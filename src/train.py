"""Treino dos modelos de previsão de falhas.

Uso pela linha de comando (a partir da raiz do projeto):

    python -m src.train

O que acontece aqui (tudo SÓ com o conjunto de treino — o teste fica guardado
para a avaliação final em ``src/evaluate.py``):

1. Baseline: regressão logística (dados padronizados, ``class_weight='balanced'``).
2. Modelo principal: XGBoost com ``scale_pos_weight`` = negativos/positivos do treino
   e uma busca de hiperparâmetros pequena, com validação cruzada estratificada de
   5 dobras, otimizando PR-AUC (average precision).
3. Escolha do limiar de alarme: previsões *out-of-fold* da validação cruzada no
   treino e o limiar que maximiza F2 (recall pesa mais que precisão, porque deixar
   passar uma falha custa mais caro que uma parada desnecessária).
4. Salva em ``models/``: ``xgb_model.json`` (formato nativo do XGBoost, versionado),
   ``threshold.json`` (limiar + configuração), ``metrics.json`` (métricas de CV) e
   ``baseline_logreg.joblib`` (fora do Git).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import PROJECT_ROOT
from src.features import PROCESSED_DIR, RANDOM_STATE, TYPE_MAPPING, load_processed, main as build_features

MODELS_DIR = PROJECT_ROOT / "models"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.json"
BASELINE_PATH = MODELS_DIR / "baseline_logreg.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

N_FOLDS = 5
N_ITER_SEARCH = 20  # busca pequena e honesta: 20 combinações x 5 dobras = 100 treinos
F_BETA = 2          # F2: recall vale "o dobro" da precisão

# Espaço de busca modesto (árvores rasas e poucas rodadas mantêm o .json pequeno)
PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
}


def get_cv() -> StratifiedKFold:
    """Validação cruzada estratificada usada em todo o projeto (mesmas dobras sempre)."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def make_baseline() -> "sklearn.pipeline.Pipeline":
    """Regressão logística com padronização e peso maior para a classe rara."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
    )


def make_xgb(scale_pos_weight: float, **params) -> xgb.XGBClassifier:
    """XGBoost configurado para classe desbalanceada."""
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,  # paralelismo fica na busca (n_jobs=-1 abaixo)
        **params,
    )


def best_f_beta_threshold(y_true, proba, beta: float = F_BETA) -> dict:
    """Limiar que maximiza F-beta, calculado sobre previsões out-of-fold."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall têm um elemento a mais que thresholds (ponto final sem limiar)
    p, r = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1 + beta**2) * p * r / (beta**2 * p + r)
    f = np.nan_to_num(f)
    i = int(np.argmax(f))
    return {
        "threshold": float(thresholds[i]),
        f"f{beta:g}_oof": float(f[i]),
        "precision_oof": float(p[i]),
        "recall_oof": float(r[i]),
    }


def oof_summary(y_true, proba, threshold: float) -> dict:
    """Métricas out-of-fold no treino para um limiar."""
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc_oof": float(average_precision_score(y_true, proba)),
        f"f{F_BETA}_oof_at_threshold": float(fbeta_score(y_true, pred, beta=F_BETA)),
    }


def load_xgb_model(path: Path = XGB_MODEL_PATH) -> xgb.XGBClassifier:
    """Carrega o XGBoost salvo (usado pela avaliação, pela explicação e pelo app)."""
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def load_threshold_config(path: Path = THRESHOLD_PATH) -> dict:
    """Lê o limiar de alarme e a configuração salvos no treino."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if not (PROCESSED_DIR / "X_train.csv").exists():
        print("[treino] data/processed/ não encontrado — gerando com src.features ...")
        build_features()

    X_train, _, y_train, _ = load_processed()  # o teste NÃO é usado aqui
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw = n_neg / n_pos
    cv = get_cv()
    print(f"[treino] {len(y_train)} linhas | falhas: {n_pos} | scale_pos_weight = {spw:.2f}")

    # 1) Baseline -------------------------------------------------------------
    baseline = make_baseline()
    proba_base_oof = cross_val_predict(baseline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr_base = best_f_beta_threshold(y_train, proba_base_oof)
    baseline.fit(X_train, y_train)
    joblib.dump(baseline, BASELINE_PATH)
    print(f"[treino] Baseline PR-AUC (OOF): {average_precision_score(y_train, proba_base_oof):.4f}")

    # 2) XGBoost + busca de hiperparâmetros ------------------------------------
    search = RandomizedSearchCV(
        make_xgb(spw),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X_train, y_train)
    best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    i_best = int(search.best_index_)
    cv_mean = float(search.cv_results_["mean_test_score"][i_best])
    cv_std = float(search.cv_results_["std_test_score"][i_best])
    print(f"[treino] Melhores hiperparâmetros: {best_params}")
    print(f"[treino] XGBoost PR-AUC na CV: {cv_mean:.4f} ± {cv_std:.4f}")

    # 3) Limiar pelo F2 nas previsões out-of-fold -------------------------------
    xgb_best = make_xgb(spw, **best_params)
    proba_xgb_oof = cross_val_predict(xgb_best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr_xgb = best_f_beta_threshold(y_train, proba_xgb_oof)
    print(
        f"[treino] Limiar escolhido (máx. F2 OOF): {thr_xgb['threshold']:.4f} | "
        f"recall OOF {thr_xgb['recall_oof']:.3f} | precisão OOF {thr_xgb['precision_oof']:.3f}"
    )

    # 4) Modelo final em todo o treino e arquivos -------------------------------
    xgb_best.fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    xgb_best.save_model(XGB_MODEL_PATH)

    config = {
        "model_file": XGB_MODEL_PATH.name,
        "threshold": round(thr_xgb["threshold"], 4),
        "default_threshold": 0.5,
        "threshold_method": (
            f"Máximo F{F_BETA} nas previsões out-of-fold da validação cruzada "
            f"estratificada ({N_FOLDS} dobras) no conjunto de treino. O teste não foi usado."
        ),
        "baseline_threshold": round(thr_base["threshold"], 4),
        "features": list(X_train.columns),
        "type_mapping": TYPE_MAPPING,
        "target": y_train.name,
    }
    _write_json(THRESHOLD_PATH, config)

    metrics = {
        "gerado_em_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versoes": {"xgboost": xgb.__version__, "scikit-learn": sklearn.__version__},
        "treino": {
            "linhas": int(len(y_train)),
            "falhas": n_pos,
            "scale_pos_weight": round(spw, 4),
            "cv": f"StratifiedKFold(n_splits={N_FOLDS}, shuffle=True, random_state={RANDOM_STATE})",
        },
        "busca_hiperparametros": {
            "metodo": f"RandomizedSearchCV, {N_ITER_SEARCH} combinações, scoring=average_precision",
            "espaco": PARAM_DISTRIBUTIONS,
            "melhores_parametros": best_params,
            "pr_auc_cv_media": round(cv_mean, 4),
            "pr_auc_cv_desvio": round(cv_std, 4),
        },
        "cv_oof": {
            "baseline": {**{k: round(v, 4) for k, v in thr_base.items()},
                         **{k: round(v, 4) for k, v in oof_summary(y_train, proba_base_oof, thr_base["threshold"]).items()}},
            "xgboost": {**{k: round(v, 4) for k, v in thr_xgb.items()},
                        **{k: round(v, 4) for k, v in oof_summary(y_train, proba_xgb_oof, thr_xgb["threshold"]).items()}},
        },
    }
    # Preserva resultados de teste de uma avaliação anterior, se houver
    if METRICS_PATH.exists():
        old = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if "teste" in old:
            metrics["teste"] = old["teste"]
    _write_json(METRICS_PATH, metrics)

    size_kb = XGB_MODEL_PATH.stat().st_size / 1024
    print(f"[treino] Modelo salvo em {XGB_MODEL_PATH.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")
    print(f"[treino] Configuração em {THRESHOLD_PATH.relative_to(PROJECT_ROOT)} e métricas de CV em {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
