"""Painel de manutenção preditiva (Streamlit).

Rodar localmente, a partir da raiz do projeto:

    streamlit run app/streamlit_app.py

No Streamlit Community Cloud, o arquivo principal é ``app/streamlit_app.py``.
O app só lê arquivos versionados (``models/xgb_model.json``, ``models/threshold.json``,
``models/metrics.json`` e ``reports/figures/``): não baixa dados nem retreina nada.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# Raiz do repositório no sys.path, para importar o pacote src/ (funciona local e na nuvem)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explain import FEATURE_LABELS, explain_row, get_explainer  # noqa: E402
from src.features import build_feature_row  # noqa: E402

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

st.set_page_config(page_title="Manutenção Preditiva com IA", page_icon="🔧", layout="wide")


# ----------------------------------------------------------------------------- dados do app

@st.cache_data
def load_json(nome: str) -> dict:
    return json.loads((MODELS_DIR / nome).read_text(encoding="utf-8"))


@st.cache_resource
def warm_up_model():
    """Carrega o XGBoost e o explicador SHAP uma única vez por servidor."""
    return get_explainer()


CONFIG = load_json("threshold.json")
METRICS = load_json("metrics.json")
THRESHOLD = float(CONFIG["threshold"])
warm_up_model()

# Faixas dos controles = mínimo e máximo observados no dataset AI4I 2020;
# valores padrão = medianas do dataset.
SENSORES = {
    "air": dict(label="Temperatura do ar [K]", min=295.0, max=305.0, default=300.1, step=0.1),
    "process": dict(label="Temperatura de processo [K]", min=305.5, max=314.0, default=310.1, step=0.1),
    "rpm": dict(label="Rotação [rpm]", min=1150, max=2900, default=1503, step=1),
    "torque": dict(label="Torque [Nm]", min=3.5, max=77.0, default=40.1, step=0.1),
    "wear": dict(label="Desgaste da ferramenta [min]", min=0, max=255, default=108, step=1),
}

# Cenários prontos: registros REAIS do arquivo ai4i2020.csv (UDI = número da linha no dataset)
PRESETS = {
    "✅ Máquina normal": dict(
        udi=1, type="M", air=298.1, process=308.6, rpm=1551, torque=42.8, wear=0,
        desc="Registro sem falha: ferramenta nova, torque e rotação em faixa normal."),
    "🪓 Ferramenta gasta": dict(
        udi=1683, type="H", air=297.9, process=307.4, rpm=1604, torque=36.1, wear=225,
        desc="Registro com falha por desgaste da ferramenta (TWF): 225 min de uso, demais sensores normais."),
    "🏋️ Sobrecarga": dict(
        udi=6498, type="L", air=300.8, process=309.9, rpm=1312, torque=65.3, wear=192,
        desc="Registro com falha por sobrecarga (OSF): torque alto com ferramenta já gasta."),
    "🌡️ Calor preso": dict(
        udi=4845, type="M", air=303.4, process=311.8, rpm=1316, torque=50.9, wear=69,
        desc="Registro com falha por dissipação de calor (HDF): pouca diferença de temperatura e rotação baixa."),
}


def init_state() -> None:
    if "type" not in st.session_state:
        st.session_state["type"] = "L"
        for k, s in SENSORES.items():
            st.session_state[k] = s["default"]
        st.session_state["preset"] = None


def apply_preset(nome: str) -> None:
    p = PRESETS[nome]
    st.session_state["type"] = p["type"]
    for k in SENSORES:
        st.session_state[k] = type(SENSORES[k]["default"])(p[k])
    st.session_state["preset"] = nome


def fmt_int(n: int) -> str:
    """Inteiro com separador de milhar brasileiro (2.000)."""
    return f"{int(n):,}".replace(",", ".")


def fmt(x: float, casas: int = 3) -> str:
    """Número no formato brasileiro (vírgula decimal)."""
    return f"{x:.{casas}f}".replace(".", ",")


# ----------------------------------------------------------------------------- cabeçalho

st.title("🔧 Manutenção Preditiva com IA")
st.markdown(
    "**Com essas leituras dos sensores, a máquina vai falhar? E por quê?** "
    "Modelo XGBoost treinado no dataset público AI4I 2020, com explicação SHAP de cada previsão."
)

aba_sim, aba_desemp, aba_dados = st.tabs(["🎛️ Simulador", "📊 Desempenho do modelo", "📚 Sobre os dados"])

# ----------------------------------------------------------------------------- aba 1

with aba_sim:
    init_state()
    st.markdown("Ajuste as leituras dos sensores ou carregue um **cenário real** do dataset:")
    cols = st.columns(len(PRESETS))
    for col, nome in zip(cols, PRESETS):
        col.button(nome, on_click=apply_preset, args=(nome,), width="stretch")
    if st.session_state.get("preset"):
        p = PRESETS[st.session_state["preset"]]
        st.caption(f"Cenário carregado: **{st.session_state['preset']}** — registro real UDI {p['udi']}. {p['desc']}")

    col_in, col_out = st.columns([1, 1.35], gap="large")

    with col_in:
        st.subheader("Leituras dos sensores")
        st.radio("Tipo de produto (qualidade)", ["L", "M", "H"], key="type", horizontal=True,
                 help="L = baixa, M = média, H = alta qualidade.")
        for k, s in SENSORES.items():
            st.slider(s["label"], min_value=s["min"], max_value=s["max"], step=s["step"], key=k)

        row = build_feature_row(
            st.session_state["type"], st.session_state["air"], st.session_state["process"],
            st.session_state["rpm"], st.session_state["torque"], st.session_state["wear"],
        )
        c1, c2 = st.columns(2)
        c1.metric("Potência calculada", f"{fmt_int(round(row['power_w'].iloc[0]))} W")
        c2.metric("Dif. temperatura (processo − ar)", f"{fmt(row['temp_diff_k'].iloc[0], 1)} K")
        if st.session_state["process"] <= st.session_state["air"]:
            st.warning("A temperatura de processo normalmente fica ~10 K acima da do ar. "
                       "Combinações muito fora do dataset geram previsões pouco confiáveis.")

    with col_out:
        resultado = explain_row(row)
        prob = resultado["probabilidade"]
        st.subheader("Previsão")
        m1, m2 = st.columns(2)
        m1.metric("Probabilidade de falha", f"{fmt(prob * 100, 1)}%")
        m2.metric("Limiar de alarme", fmt(THRESHOLD, 4))
        if resultado["alarme"]:
            st.error(f"🚨 **ALARME**: probabilidade acima do limiar ({fmt(THRESHOLD, 4)}). Recomenda-se inspeção.")
        else:
            st.success(f"✅ **Sem alarme**: probabilidade abaixo do limiar ({fmt(THRESHOLD, 4)}).")

        st.markdown("**Por que o modelo chegou nesse valor?** (contribuição SHAP de cada variável)")
        plt.figure()
        shap.plots.waterfall(resultado["explanation"], max_display=8, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 4.6)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close("all")

        top = resultado["shap"].head(3)
        frases = []
        for feat, val in top.items():
            direcao = "aumenta" if val > 0 else "reduz"
            frases.append(f"**{FEATURE_LABELS[feat]}** = {fmt(float(row[feat].iloc[0]), 1)} {direcao} o risco")
        st.markdown("Principais fatores: " + "; ".join(frases) + ".")
        st.caption("Barras vermelhas empurram para falha, azuis para operação normal. Valores em log-odds "
                   "(escala interna do modelo); E[f(X)] é o ponto de partida médio e f(x) o resultado desta entrada.")

# ----------------------------------------------------------------------------- aba 2

with aba_desemp:
    teste = METRICS["teste"]
    st.subheader("Resultado no conjunto de teste")
    st.markdown(
        f"Avaliação em **{fmt_int(teste['linhas'])} registros nunca vistos no treino**, com **{teste['falhas']} falhas**. "
        "Cada modelo usa o seu limiar escolhido no treino."
    )

    linhas = []
    for nome, chave in [("Regressão logística (baseline)", "baseline"), ("XGBoost (modelo final)", "xgboost")]:
        m = teste[chave]["limiar_escolhido"]
        linhas.append({
            "Modelo": nome, "Limiar": fmt(m["limiar"], 4), "Recall": fmt(m["recall"]),
            "Precisão": fmt(m["precisao"]), "F1": fmt(m["f1"]), "PR-AUC": fmt(m["pr_auc"]),
            "ROC-AUC": fmt(m["roc_auc"]), "Falhas detectadas": m["falhas_detectadas"],
            "Alarmes falsos": m["alarmes_falsos"],
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")

    ing = teste["modelo_ingenuo"]
    xgb_m = teste["xgboost"]["limiar_escolhido"]
    ic = teste["ic95_xgboost_limiar_escolhido"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Falhas detectadas (XGBoost)", xgb_m["falhas_detectadas"])
    k2.metric("Alarmes falsos (XGBoost)", xgb_m["alarmes_falsos"])
    k3.metric("Recall — IC 95% (bootstrap)", f"{fmt(ic['recall'][0])} a {fmt(ic['recall'][1])}")
    st.info(
        f"**Por que não acurácia?** Um modelo que responde sempre \"não vai falhar\" tem acurácia de "
        f"**{fmt(ing['acuracia'] * 100, 1)}%** e detecta **{ing['falhas_detectadas']}** falhas (recall "
        f"{fmt(ing['recall'], 0)}). Por isso o foco está em recall, precisão e PR-AUC."
    )

    f1c, f2c = st.columns(2)
    f1c.image(str(FIGURES_DIR / "matriz_confusao_xgboost.png"), caption="Matriz de confusão do XGBoost no teste")
    f2c.image(str(FIGURES_DIR / "curva_precisao_recall.png"), caption="Curva precisão × recall: baseline × XGBoost")

    st.subheader("O limiar de alarme é uma decisão de manutenção")
    st.markdown(
        f"""
O modelo entrega uma **probabilidade**; o **limiar** é a regra que dispara o alarme — como o *setpoint* de um alarme no supervisório.

- **Limiar mais baixo** → pega mais falhas, mas para a máquina à toa com mais frequência.
- **Limiar mais alto** → menos paradas desnecessárias, mas mais falhas passam sem aviso.

Aqui o limiar **{fmt(THRESHOLD, 4)}** foi escolhido **só com dados de treino**: previsões *out-of-fold* da validação
cruzada e o valor que maximiza o **F2** (recall pesa mais que precisão, porque uma falha não detectada custa mais caro
que uma inspeção). O teste não foi usado nessa escolha. Numa fábrica real, o ideal é definir o limiar pelo custo de cada erro.
"""
    )
    st.image(str(FIGURES_DIR / "limiar_precisao_recall.png"), caption="Recall e precisão do XGBoost no teste para cada limiar")

    with st.expander("Como o modelo foi treinado"):
        busca = METRICS["busca_hiperparametros"]
        st.markdown(
            f"- Treino: {fmt_int(METRICS['treino']['linhas'])} registros ({METRICS['treino']['falhas']} falhas); divisão 80/20 estratificada, semente 42.\n"
            + f"- `scale_pos_weight` = {fmt(METRICS['treino']['scale_pos_weight'], 2)} para compensar o desbalanceamento.\n"
            + f"- {busca['metodo']}.\n"
            + f"- PR-AUC na validação cruzada: {fmt(busca['pr_auc_cv_media'])} ± {fmt(busca['pr_auc_cv_desvio'])}.\n"
            + f"- Melhores hiperparâmetros: `{busca['melhores_parametros']}`"
        )
        st.image(str(FIGURES_DIR / "shap_importancia.png"), caption="Importância global das variáveis (média |SHAP| no teste)")

# ----------------------------------------------------------------------------- aba 3

with aba_dados:
    st.subheader("AI4I 2020 Predictive Maintenance Dataset")
    st.markdown(
        """
Conjunto **sintético** publicado no UCI Machine Learning Repository para reproduzir dados de manutenção preditiva
de uma máquina de usinagem.

- **10.000 registros**, 339 com falha (**3,39%**) — desbalanceado, como numa fábrica.
- **Entradas do modelo:** tipo de produto (L/M/H), temperatura do ar, temperatura de processo, rotação, torque e
  desgaste da ferramenta, mais duas variáveis físicas criadas no projeto: **potência** (torque × rotação em rad/s) e
  **diferença de temperatura** (processo − ar).
- **Fora do modelo:** identificadores e as colunas de modo de falha (TWF, HDF, PWF, OSF, RNF), que só são conhecidas
  depois da falha e vazariam a resposta.
"""
    )
    st.subheader("Limitações")
    st.markdown(
        """
- **Dados sintéticos:** o modelo demonstra o método; não substitui validação com dados de uma máquina real.
- **Falha por desgaste da ferramenta é, em boa parte, sorteio:** pela descrição do dataset, a ferramenta falha ou é
  trocada num momento aleatório entre ~200 e 240 min de uso. Os sensores mostram a zona de risco, mas não o instante
  da quebra — 9 das 12 falhas não detectadas no teste são desse tipo.
- **Falha de potência é uma regra fixa** (fora de 3.500–9.000 W sempre falha), fácil para o modelo aprender.
- **Sem dimensão de tempo:** cada linha é um instante isolado; não há tendência dos sensores.
- **Poucas falhas no teste (68):** as métricas têm incerteza (veja o intervalo de confiança na aba de desempenho).
"""
    )
    st.subheader("Fonte e licença")
    st.markdown(
        """
- AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository.
  [https://doi.org/10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)
- Licença dos dados: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Artigo: S. Matzka, *Explainable Artificial Intelligence for Predictive Maintenance Applications*, AI4I 2020.
- Código do projeto: licença MIT — [repositório no GitHub](https://github.com/DenisPaulo/manutencao-preditiva-ia).
"""
    )
