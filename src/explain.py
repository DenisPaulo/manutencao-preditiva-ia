"""Explicação do modelo com SHAP (por que o modelo deu alarme?).

Uso pela linha de comando (a partir da raiz do projeto, depois de ``python -m src.train``):

    python -m src.explain

Gera em ``reports/figures/``:
- visão global: ``shap_beeswarm.png`` e ``shap_importancia.png`` (média de |SHAP|);
- visão local (casos reais do teste): ``shap_waterfall_falha_detectada.png``,
  ``shap_waterfall_falha_nao_detectada.png`` e ``shap_waterfall_alarme_falso.png``.

A função :func:`explain_row` devolve os valores SHAP de uma única entrada e é
reaproveitada pelo app.

Os valores SHAP do XGBoost estão na escala de *log-odds* (logaritmo da chance de
falha): valores positivos empurram a previsão para "falha", negativos para
"operação normal".
"""

from __future__ import annotations

from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data import PROJECT_ROOT, load_data
from src.evaluate import FIGURES_DIR
from src.features import get_train_test, load_processed
from src.train import load_threshold_config, load_xgb_model

# Nomes amigáveis (pt-BR) para os gráficos
FEATURE_LABELS = {
    "type": "Tipo de produto (L=0, M=1, H=2)",
    "air_temperature_k": "Temperatura do ar [K]",
    "process_temperature_k": "Temperatura de processo [K]",
    "rotational_speed_rpm": "Rotação [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Desgaste da ferramenta [min]",
    "power_w": "Potência [W]",
    "temp_diff_k": "Dif. temperatura processo-ar [K]",
}
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


@lru_cache(maxsize=1)
def load_xgb_model_cached():
    """Modelo XGBoost carregado uma única vez (evita reler o .json a cada chamada)."""
    return load_xgb_model()


@lru_cache(maxsize=1)
def get_explainer() -> shap.TreeExplainer:
    """TreeExplainer do XGBoost final (criado uma vez e reaproveitado)."""
    return shap.TreeExplainer(load_xgb_model_cached())


def explain_row(row: pd.DataFrame | pd.Series | dict) -> dict:
    """Valores SHAP de UMA entrada (formato das features do modelo).

    Aceita um DataFrame de 1 linha, uma Series ou um dict com as colunas de
    ``models/threshold.json -> features`` (use ``src.features.build_feature_row``
    para montar a partir das leituras dos sensores).

    Retorna um dict com:
    - ``probabilidade``: probabilidade de falha prevista;
    - ``alarme``: True se a probabilidade ≥ limiar escolhido;
    - ``valor_base``: valor esperado do modelo (log-odds) antes de ver a entrada;
    - ``shap``: Series (log-odds) por feature, ordenada por |impacto|;
    - ``explanation``: objeto ``shap.Explanation`` pronto para ``shap.plots.waterfall``.
    """
    config = load_threshold_config()
    if isinstance(row, dict):
        row = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        row = row.to_frame().T
    row = row[config["features"]].astype(float)
    if len(row) != 1:
        raise ValueError(f"explain_row espera exatamente 1 linha, recebeu {len(row)}.")

    explainer = get_explainer()
    expl = explainer(row)[0]
    expl.feature_names = [FEATURE_LABELS.get(f, f) for f in config["features"]]
    proba = float(load_xgb_model_cached().predict_proba(row)[:, 1][0])
    valores = pd.Series(expl.values, index=config["features"])
    return {
        "probabilidade": proba,
        "alarme": proba >= float(config["threshold"]),
        "valor_base": float(expl.base_values),
        "shap": valores.reindex(valores.abs().sort_values(ascending=False).index),
        "explanation": expl,
    }


def compute_test_shap() -> tuple[shap.Explanation, pd.DataFrame]:
    """SHAP para todo o conjunto de teste + tabela de contexto de cada linha.

    A tabela traz o UDI original, os modos de falha reais (só para interpretar os
    casos — o modelo nunca viu essas colunas), a probabilidade e a previsão.
    """
    config = load_threshold_config()
    _, X_test, _, y_test = load_processed()
    X_test = X_test[config["features"]]

    # Recria a divisão para recuperar o índice original (UDI e modos de falha)
    df_raw = load_data()
    _, X_test_idx, _, _ = get_train_test(df_raw)
    assert np.allclose(X_test_idx[config["features"]].to_numpy(), X_test.to_numpy()), \
        "data/processed/ desatualizado — rode `python -m src.features`."

    model = load_xgb_model_cached()
    proba = model.predict_proba(X_test)[:, 1]
    thr = float(config["threshold"])

    contexto = df_raw.loc[X_test_idx.index, ["UDI", "Type"] + FAILURE_MODES].reset_index(drop=True)
    contexto["real"] = y_test.to_numpy()
    contexto["probabilidade"] = proba
    contexto["previsto"] = (proba >= thr).astype(int)

    expl = get_explainer()(X_test)
    expl.feature_names = [FEATURE_LABELS.get(f, f) for f in config["features"]]
    return expl, contexto


def select_cases(contexto: pd.DataFrame) -> dict[str, int | None]:
    """Escolhe 3 casos reais do teste com regras fixas (sem escolha manual):

    - falha detectada: verdadeiro positivo com MAIOR probabilidade;
    - falha não detectada: falso negativo com MENOR probabilidade (o caso mais "invisível");
    - alarme falso: falso positivo com MAIOR probabilidade (o alarme mais convicto).
    """
    real, prev = contexto["real"], contexto["previsto"]
    grupos = {
        "falha_detectada": (contexto[(real == 1) & (prev == 1)], "max"),
        "falha_nao_detectada": (contexto[(real == 1) & (prev == 0)], "min"),
        "alarme_falso": (contexto[(real == 0) & (prev == 1)], "max"),
    }
    casos = {}
    for nome, (sub, modo) in grupos.items():
        if sub.empty:
            casos[nome] = None
        else:
            casos[nome] = int(sub["probabilidade"].idxmax() if modo == "max" else sub["probabilidade"].idxmin())
    return casos


def plot_beeswarm(expl: shap.Explanation):
    plt.figure()
    shap.plots.beeswarm(expl, max_display=10, show=False, color_bar_label="Valor da variável")
    fig = plt.gcf()
    # Traduz os rótulos "Low"/"High" da barra de cores
    for ax in fig.axes[1:]:
        labels = [t.get_text() for t in ax.get_yticklabels()]
        if labels == ["Low", "High"]:
            ax.set_yticks(ax.get_yticks(), ["Baixo", "Alto"])
    fig.set_size_inches(9, 5.5)
    plt.title("SHAP — impacto de cada variável em cada máquina do teste")
    plt.xlabel("Valor SHAP (log-odds): → aumenta o risco de falha | ← diminui")
    plt.tight_layout()
    return fig


def plot_importance(expl: shap.Explanation):
    plt.figure()
    shap.plots.bar(expl, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5)
    plt.title("Importância global: média de |SHAP| no teste")
    plt.xlabel("Média de |valor SHAP| (log-odds)")
    plt.tight_layout()
    return fig


def plot_waterfall(expl_row: shap.Explanation, titulo: str):
    plt.figure()
    shap.plots.waterfall(expl_row, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5.5)
    plt.title(titulo, fontsize=11)
    plt.tight_layout()
    return fig


def mean_abs_shap(expl: shap.Explanation) -> pd.Series:
    """Importância global (média de |SHAP|) em ordem decrescente."""
    return pd.Series(np.abs(expl.values).mean(axis=0), index=expl.feature_names).sort_values(ascending=False)


def describe_case(contexto: pd.DataFrame, i: int) -> str:
    c = contexto.loc[i]
    modos = [m for m in FAILURE_MODES if c[m] == 1]
    return (f"UDI {int(c['UDI'])} (tipo {c['Type']}) — probabilidade {c['probabilidade']:.1%}, "
            f"real={'falha' if c['real'] else 'sem falha'}, modo(s): {', '.join(modos) or 'nenhum'}")


TITULOS = {
    "falha_detectada": "Falha detectada (verdadeiro positivo)",
    "falha_nao_detectada": "Falha NÃO detectada (falso negativo)",
    "alarme_falso": "Alarme falso (falso positivo)",
}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    expl, contexto = compute_test_shap()

    for nome, fig in {"shap_beeswarm.png": plot_beeswarm(expl),
                      "shap_importancia.png": plot_importance(expl)}.items():
        fig.savefig(FIGURES_DIR / nome, dpi=110, bbox_inches="tight")
        plt.close(fig)

    print("[explicação] Importância global (média |SHAP|, log-odds):")
    for f, v in mean_abs_shap(expl).items():
        print(f"    {f:<36} {v:.3f}")

    for nome, i in select_cases(contexto).items():
        if i is None:
            print(f"[explicação] Nenhum caso de '{nome}' no teste — gráfico não gerado.")
            continue
        descricao = describe_case(contexto, i)
        fig = plot_waterfall(expl[i], f"{TITULOS[nome]}\n{descricao}")
        fig.savefig(FIGURES_DIR / f"shap_waterfall_{nome}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        top = pd.Series(expl[i].values, index=expl.feature_names)
        top = top.reindex(top.abs().sort_values(ascending=False).index)[:3]
        print(f"[explicação] {TITULOS[nome]}: {descricao}")
        print("    principais contribuições (log-odds): "
              + "; ".join(f"{k} = {expl[i].data[expl.feature_names.index(k)]:g} → {v:+.2f}" for k, v in top.items()))

    print(f"[explicação] Figuras salvas em {FIGURES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
