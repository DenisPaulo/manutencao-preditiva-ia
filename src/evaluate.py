"""Avaliação final no conjunto de teste (usado uma única vez, no fim).

Uso pela linha de comando (a partir da raiz do projeto, depois de ``python -m src.train``):

    python -m src.evaluate

Compara, no teste:
- modelo ingênuo "nunca falha" (para mostrar por que acurácia engana);
- baseline (regressão logística) e XGBoost, no limiar padrão 0,5 e no limiar
  escolhido no treino (máximo F2 out-of-fold).

Grava os resultados em ``models/metrics.json`` (seção ``teste``) e as figuras em
``reports/figures/``.
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # gera PNG sem precisar de tela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import PROJECT_ROOT
from src.features import RANDOM_STATE, load_processed
from src.train import BASELINE_PATH, METRICS_PATH, load_threshold_config, load_xgb_model

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
N_BOOTSTRAP = 2000

COR_BASE, COR_XGB = "#8C8C8C", "#C44E52"


def compute_metrics(y_true, proba, threshold: float) -> dict:
    """Métricas de classificação para um limiar de alarme."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "limiar": round(float(threshold), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "precisao": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, pred, beta=2, zero_division=0)), 4),
        "acuracia": round(float(accuracy_score(y_true, pred)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "matriz_confusao": {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)},
        "falhas_detectadas": f"{int(tp)} de {int(tp + fn)}",
        "alarmes_falsos": int(fp),
    }


def naive_metrics(y_true) -> dict:
    """Modelo ingênuo que sempre diz 'não vai falhar'."""
    y_true = np.asarray(y_true)
    pred = np.zeros_like(y_true)
    return {
        "descricao": "Sempre prevê 'sem falha'",
        "acuracia": round(float(accuracy_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "falhas_detectadas": f"0 de {int(y_true.sum())}",
    }


def bootstrap_ci(y_true, proba, threshold: float, n: int = N_BOOTSTRAP, seed: int = RANDOM_STATE) -> dict:
    """Intervalo de confiança de 95% (bootstrap percentil) para recall, precisão e PR-AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    stats = {"recall": [], "precisao": [], "pr_auc": []}
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, pb, pr = y_true[idx], proba[idx], pred[idx]
        if yt.sum() == 0:
            continue
        stats["recall"].append(recall_score(yt, pr, zero_division=0))
        stats["precisao"].append(precision_score(yt, pr, zero_division=0))
        stats["pr_auc"].append(average_precision_score(yt, pb))
    return {
        k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
        for k, v in stats.items()
    } | {"reamostragens": n, "metodo": "bootstrap percentil 95%, semente 42"}


def predict_test():
    """Carrega modelos e devolve (X_test, y_test, proba_baseline, proba_xgb, config)."""
    _, X_test, _, y_test = load_processed()
    config = load_threshold_config()
    xgb_model = load_xgb_model()
    baseline = joblib.load(BASELINE_PATH)
    proba_xgb = xgb_model.predict_proba(X_test[config["features"]])[:, 1]
    proba_base = baseline.predict_proba(X_test[config["features"]])[:, 1]
    return X_test, y_test, proba_base, proba_xgb, config


# ----------------------------------------------------------------------------- figuras

def plot_confusion_matrix(y_true, proba, threshold: float, title: str):
    cm = confusion_matrix(y_true, (proba >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    rotulos = [["Verdadeiro negativo\n(operação normal)", "Falso positivo\n(alarme falso)"],
               ["Falso negativo\n(falha não detectada)", "Verdadeiro positivo\n(falha detectada)"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{rotulos[i][j]}", ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Sem falha", "Falha"])
    ax.set_yticks([0, 1], ["Sem falha", "Falha"])
    ax.set_xlabel("Previsão do modelo")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pr_curves(y_true, proba_base, proba_xgb, thr_base: float, thr_xgb: float):
    fig, ax = plt.subplots(figsize=(7, 5))
    for nome, proba, thr, cor in [("Baseline (reg. logística)", proba_base, thr_base, COR_BASE),
                                  ("XGBoost", proba_xgb, thr_xgb, COR_XGB)]:
        p, r, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(r, p, color=cor, lw=2, label=f"{nome} — PR-AUC = {ap:.3f}")
        pred = proba >= thr
        ax.scatter(recall_score(y_true, pred), precision_score(y_true, pred, zero_division=0),
                   color=cor, s=70, edgecolor="black", zorder=3)
    prevalencia = float(np.mean(y_true))
    ax.axhline(prevalencia, color="black", ls=":", lw=1, label=f"Sorteio aleatório ({prevalencia:.1%} de falhas)")
    ax.set_xlabel("Recall (falhas detectadas / total de falhas)")
    ax.set_ylabel("Precisão (alarmes corretos / total de alarmes)")
    ax.set_title("Curva precisão x recall no teste\n(pontos = limiar de alarme escolhido no treino)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_threshold_curve(y_true, proba, threshold: float):
    grid = np.linspace(0.01, 0.99, 99)
    rec = [recall_score(y_true, proba >= t) for t in grid]
    prec = [precision_score(y_true, proba >= t, zero_division=0) for t in grid]
    f2 = [fbeta_score(y_true, proba >= t, beta=2, zero_division=0) for t in grid]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(grid, rec, lw=2, color="#4C72B0", label="Recall (falhas detectadas)")
    ax.plot(grid, prec, lw=2, color="#DD8452", label="Precisão (alarmes corretos)")
    ax.plot(grid, f2, lw=1.5, ls="--", color="#55A868", label="F2")
    ax.axvline(threshold, color="black", lw=1.2, label=f"Limiar escolhido no treino = {threshold:.3f}")
    ax.set_xlabel("Limiar de alarme (probabilidade de falha)")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.03)
    ax.set_title("XGBoost no teste: efeito do limiar de alarme")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def main() -> None:
    _, y_test, proba_base, proba_xgb, config = predict_test()
    thr_xgb = float(config["threshold"])
    thr_base = float(config["baseline_threshold"])

    resultados = {
        "linhas": int(len(y_test)),
        "falhas": int(y_test.sum()),
        "modelo_ingenuo": naive_metrics(y_test),
        "baseline": {
            "limiar_0_5": compute_metrics(y_test, proba_base, 0.5),
            "limiar_escolhido": compute_metrics(y_test, proba_base, thr_base),
        },
        "xgboost": {
            "limiar_0_5": compute_metrics(y_test, proba_xgb, 0.5),
            "limiar_escolhido": compute_metrics(y_test, proba_xgb, thr_xgb),
        },
        "modelo_final": f"XGBoost com limiar {thr_xgb}",
        "ic95_xgboost_limiar_escolhido": bootstrap_ci(y_test, proba_xgb, thr_xgb),
    }

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    metrics["teste"] = resultados
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figuras = {
        "matriz_confusao_xgboost.png": plot_confusion_matrix(
            y_test, proba_xgb, thr_xgb, f"XGBoost no teste (limiar {thr_xgb:.3f})"),
        "curva_precisao_recall.png": plot_pr_curves(y_test, proba_base, proba_xgb, thr_base, thr_xgb),
        "limiar_precisao_recall.png": plot_threshold_curve(y_test, proba_xgb, thr_xgb),
    }
    for nome, fig in figuras.items():
        fig.savefig(FIGURES_DIR / nome, dpi=110, bbox_inches="tight")
        plt.close(fig)

    # Resumo no terminal
    ing = resultados["modelo_ingenuo"]
    print(f"[avaliação] Teste: {resultados['linhas']} linhas, {resultados['falhas']} falhas")
    print(f"[avaliação] Modelo ingênuo 'nunca falha': acurácia {ing['acuracia']:.2%} | recall {ing['recall']:.0%}")
    linhas = []
    for modelo in ("baseline", "xgboost"):
        for chave, m in resultados[modelo].items():
            cm = m["matriz_confusao"]
            linhas.append({
                "modelo": modelo, "limiar": m["limiar"], "recall": m["recall"], "precisao": m["precisao"],
                "f1": m["f1"], "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                "VP": cm["vp"], "FN": cm["fn"], "FP": cm["fp"], "VN": cm["vn"],
            })
    print(pd.DataFrame(linhas).to_string(index=False))
    ic = resultados["ic95_xgboost_limiar_escolhido"]
    print(f"[avaliação] IC95% (bootstrap) XGBoost: recall {ic['recall']} | precisão {ic['precisao']} | PR-AUC {ic['pr_auc']}")
    print(f"[avaliação] Figuras em {FIGURES_DIR.relative_to(PROJECT_ROOT)}/ e métricas em {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
