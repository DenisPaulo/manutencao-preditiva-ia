"""Painel de manutenção preditiva (Streamlit) — V1 (AI4I 2020) e V2 (NASA C-MAPSS RUL).

Rodar localmente, a partir da raiz do projeto:

    streamlit run app/streamlit_app.py

No Streamlit Community Cloud, o arquivo principal é ``app/streamlit_app.py``.
O app só lê arquivos versionados (``models/*.json`` e ``reports/figures/``):
não baixa dados brutos nem retreina nada em produção.
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

from src.cmapss import (  # noqa: E402
    ATTENTION_ZONE,
    CRITICAL_ZONE,
    RUL_CAP,
    explain_rul_row,
    feature_label,
    get_rul_explainer,
)
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
def warm_up_models():
    """Carrega os modelos XGBoost (V1 e V2) e os explicadores SHAP uma única vez."""
    return get_explainer(), get_rul_explainer()


CONFIG = load_json("threshold.json")
METRICS = load_json("metrics.json")
RUL_METRICS = load_json("rul_metrics.json")
RUL_SAMPLES = load_json("rul_sample_engines.json")
THRESHOLD = float(CONFIG["threshold"])
warm_up_models()

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
    "**V1 — Vai falhar? E por quê?** (classificação no dataset AI4I 2020) · "
    "**V2 — Quantos ciclos restam até a falha?** (regressão de vida útil remanescente / RUL no NASA C-MAPSS FD001), "
    "ambos com **XGBoost** e explicação **SHAP** de cada previsão."
)

aba_sim, aba_rul, aba_desemp, aba_dados = st.tabs([
    "🎛️ V1 — Simulador de falha (AI4I)",
    "⏱️ V2 — Vida útil / RUL (NASA C-MAPSS)",
    "📊 Desempenho dos modelos",
    "📚 Sobre os dados",
])

# ----------------------------------------------------------------------------- aba 1 (V1)

with aba_sim:
    init_state()
    st.markdown("Ajuste as leituras dos sensores ou carregue um **cenário real** do dataset AI4I 2020:")
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

# ----------------------------------------------------------------------------- aba 2 (V2 — RUL C-MAPSS)

with aba_rul:
    st.subheader("Previsão de Vida Útil Remanescente (RUL) — NASA C-MAPSS FD001")
    st.markdown(
        "Selecione um motor real do conjunto de teste **FD001** para inspecionar a trajetória de degradação "
        "ciclo a ciclo, a **vida útil remanescente prevista (em ciclos)** e quais sensores puxaram a previsão no último ciclo."
    )

    motores = RUL_SAMPLES["motores_destaque"]
    opcoes_id = list(motores.keys())
    rotulos_map = {k: motores[k]["rotulo"] for k in opcoes_id}
    motor_key = st.selectbox(
        "Motor do conjunto de teste FD001",
        options=opcoes_id,
        format_func=lambda k: rotulos_map[k],
    )
    eng = motores[motor_key]
    rul_res = explain_rul_row(eng["ultimo_ciclo_features"])
    rul_pred = rul_res["rul_previsto"]
    rul_real = eng["rul_real_final"]
    zona = rul_res["zona"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ciclos observados", f"{eng['ciclos_observados']} ciclos")
    k2.metric("RUL previsto (XGBoost)", f"{fmt(rul_pred, 1)} ciclos")
    k3.metric("RUL real (FD001)", f"{fmt(rul_real, 1)} ciclos")
    k4.metric("Erro (previsto − real)", f"{fmt(rul_pred - rul_real, 1)} ciclos")

    if zona == "Crítico":
        st.error(
            f"🚨 **ZONA CRÍTICA (≤ {CRITICAL_ZONE} ciclos)**: restam cerca de **{fmt(rul_pred, 1)} ciclos**. "
            "Recomenda-se intervenção imediata antes da falha funcional."
        )
    elif zona == "Atenção":
        st.warning(
            f"⚠️ **ZONA DE ATENÇÃO ({CRITICAL_ZONE + 1} a {ATTENTION_ZONE} ciclos)**: restam cerca de "
            f"**{fmt(rul_pred, 1)} ciclos**. Programar ordem de serviço preventiva na próxima janela."
        )
    else:
        st.success(
            f"✅ **ZONA SAUDÁVEL (> {ATTENTION_ZONE} ciclos)**: restam cerca de **{fmt(rul_pred, 1)} ciclos**. "
            "Equipamento em operação normal; manter monitoramento por condição."
        )

    col_deg, col_shap = st.columns([1.05, 1.25], gap="large")

    with col_deg:
        st.markdown("**Trajetória de degradação ao longo dos ciclos**")
        tr = eng["trajetoria"]
        fig_d, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
        ax1.axhspan(0, CRITICAL_ZONE, color="#FDECEA", alpha=0.8, label="Crítico (≤ 20)")
        ax1.axhspan(CRITICAL_ZONE, ATTENTION_ZONE, color="#FFF8E1", alpha=0.8, label="Atenção (21–50)")
        ax1.plot(tr["cycle"], tr["rul_real"], "k--", lw=1.6, label="RUL real (teto 125)")
        ax1.plot(tr["cycle"], tr["rul_pred"], color="#C44E52", lw=2.1, label="RUL previsto")
        ax1.set_ylabel("RUL [ciclos]")
        ax1.set_ylim(0, RUL_CAP + 8)
        ax1.grid(alpha=0.3)
        ax1.legend(loc="lower left", fontsize=8)

        ax2.plot(tr["cycle"], tr["s11"], color="#1F4E78", lw=1.6, label="s11 Pressão estática HPC [psia]")
        ax2_t = ax2.twinx()
        ax2_t.plot(tr["cycle"], tr["s4"], color="#DD8452", lw=1.5, ls="--", label="s4 Temp. saída LPT [°R]")
        ax2.set_xlabel("Ciclo de operação")
        ax2.set_ylabel("s11 [psia]", color="#1F4E78")
        ax2_t.set_ylabel("s4 [°R]", color="#DD8452")
        ax2.grid(alpha=0.3)
        fig_d.tight_layout()
        st.pyplot(fig_d, clear_figure=True)
        plt.close(fig_d)
        st.caption(
            "Acima: RUL previsto caindo conforme a degradação avança. Abaixo: subida da pressão estática no HPC (s11) "
            "e da temperatura de saída da LPT (s4) ao longo dos ciclos."
        )

    with col_shap:
        st.markdown("**Quais sensores reduziram ou aumentaram a RUL neste ciclo?** (SHAP local em ciclos)")
        plt.figure()
        shap.plots.waterfall(rul_res["explanation"], max_display=9, show=False)
        fig_s = plt.gcf()
        fig_s.set_size_inches(8.2, 4.8)
        plt.tight_layout()
        st.pyplot(fig_s, clear_figure=True)
        plt.close("all")

        top_rul = rul_res["shap"].head(3)
        frases_rul = []
        for feat, val in top_rul.items():
            acao = "reduz" if val < 0 else "aumenta"
            frases_rul.append(f"**{feature_label(feat)}** ({acao} a RUL em {fmt(abs(float(val)), 1)} ciclos)")
        st.markdown("Principais contribuições no último ciclo: " + "; ".join(frases_rul) + ".")
        st.caption(
            "Na regressão de RUL, os valores SHAP já estão na própria unidade do alvo (**ciclos de vida útil**): "
            "barras azuis (negativas) puxam a vida útil para baixo; barras vermelhas (positivas) indicam condição saudável."
        )

    with st.expander("Ver tabela dos 100 motores de teste do FD001"):
        filtro_zona = st.radio(
            "Filtrar por zona de manutenção prevista",
            ["Todos", "Crítico", "Atenção", "Saudável"],
            horizontal=True,
        )
        df_100 = pd.DataFrame(RUL_SAMPLES["resumo_teste"])
        if filtro_zona != "Todos":
            df_100 = df_100[df_100["zona"] == filtro_zona]
        st.dataframe(
            df_100.rename(columns={
                "unit_id": "Motor",
                "ciclos_observados": "Ciclos observados",
                "rul_real": "RUL real [ciclos]",
                "rul_previsto": "RUL previsto [ciclos]",
                "erro_ciclos": "Erro [ciclos]",
                "zona": "Zona",
            }),
            hide_index=True,
            width="stretch",
        )

# ----------------------------------------------------------------------------- aba 3 (Desempenho V1 e V2)

with aba_desemp:
    st.subheader("V2 — Regressão de Vida Útil Remanescente (NASA C-MAPSS FD001)")
    t_rul = RUL_METRICS["teste"]
    b_rul = t_rul["baseline_ridge"]
    x_rul = t_rul["xgboost_regressor"]
    st.markdown(
        f"Avaliação oficial no último ciclo observado dos **{t_rul['motores_avaliados']} motores de teste** "
        f"({fmt_int(t_rul['linhas_totais'])} ciclos no arquivo de teste), com teto linear por partes em **{RUL_CAP} ciclos**."
    )
    df_rul_m = pd.DataFrame([
        {
            "Modelo": "Ridge (baseline linear)",
            "RMSE [ciclos]": fmt(b_rul["rmse_ciclos"], 2),
            "MAE [ciclos]": fmt(b_rul["mae_ciclos"], 2),
            "R²": fmt(b_rul["r2"], 4),
            "NASA Score (menor = melhor)": fmt(b_rul["nasa_score"], 1),
        },
        {
            "Modelo": "XGBRegressor (modelo principal)",
            "RMSE [ciclos]": fmt(x_rul["rmse_ciclos"], 2),
            "MAE [ciclos]": fmt(x_rul["mae_ciclos"], 2),
            "R²": fmt(x_rul["r2"], 4),
            "NASA Score (menor = melhor)": fmt(x_rul["nasa_score"], 1),
        },
    ])
    st.dataframe(df_rul_m, hide_index=True, width="stretch")

    rk1, rk2, rk3, rk4 = st.columns(4)
    rk1.metric("RMSE (XGBoost)", f"{fmt(x_rul['rmse_ciclos'], 2)} ciclos")
    rk2.metric("MAE (XGBoost)", f"{fmt(x_rul['mae_ciclos'], 2)} ciclos")
    rk3.metric("R² no teste", fmt(x_rul["r2"], 4))
    rk4.metric("NASA Score total", fmt(x_rul["nasa_score"], 1))

    st.image(
        str(FIGURES_DIR / "rul_previsto_vs_real.png"),
        caption="NASA C-MAPSS FD001: RUL previsto vs. real e acompanhamento dos 100 motores de teste",
    )
    rc1, rc2 = st.columns(2)
    rc1.image(
        str(FIGURES_DIR / "rul_degradacao_motor.png"),
        caption="Curva de degradação de um motor e função de pontuação assimétrica da NASA",
    )
    rc2.image(
        str(FIGURES_DIR / "rul_shap_importancia.png"),
        caption="Importância global SHAP (em ciclos) no conjunto de teste FD001",
    )

    st.divider()
    teste = METRICS["teste"]
    st.subheader("V1 — Classificação de falhas (AI4I 2020)")
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

    with st.expander("Como o modelo V1 foi treinado"):
        busca = METRICS["busca_hiperparametros"]
        st.markdown(
            f"- Treino: {fmt_int(METRICS['treino']['linhas'])} registros ({METRICS['treino']['falhas']} falhas); divisão 80/20 estratificada, semente 42.\n"
            + f"- `scale_pos_weight` = {fmt(METRICS['treino']['scale_pos_weight'], 2)} para compensar o desbalanceamento.\n"
            + f"- {busca['metodo']}.\n"
            + f"- PR-AUC na validação cruzada: {fmt(busca['pr_auc_cv_media'])} ± {fmt(busca['pr_auc_cv_desvio'])}.\n"
            + f"- Melhores hiperparâmetros: `{busca['melhores_parametros']}`"
        )
        st.image(str(FIGURES_DIR / "shap_importancia.png"), caption="Importância global das variáveis (média |SHAP| no teste)")

# ----------------------------------------------------------------------------- aba 4 (Sobre os dados)

with aba_dados:
    st.subheader("V2 — NASA C-MAPSS Turbofan Engine Degradation (FD001)")
    st.markdown(
        """
Simulação de degradação até a falha (*run-to-failure*) de motores turbofan publicada pela **NASA** (Prognostics Data Repository).

- **Subconjunto FD001:** 100 motores de treino (20.631 ciclos, operando até a falha do compressor de alta pressão — HPC) e 100 motores de teste (13.096 ciclos, interrompidos antes da falha, com o gabarito oficial em `RUL_FD001.txt`).
- **Seleção de sensores:** removidos os ajustes e sensores constantes no FD001 (`op_setting_3`, `s1`, `s5`, `s6`, `s10`, `s16`, `s18`, `s19`) e mantidos os **14 sensores informativos** (`s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`).
- **Engenharia de variáveis temporais:** para cada sensor são calculados a média móvel de 5 ciclos, o desvio-padrão móvel de 5 ciclos e o deslocamento ($\Delta$) em relação aos 5 primeiros ciclos saudáveis do próprio motor.
- **Alvo (piecewise linear RUL):** limitado em **125 ciclos**, convenção clássica da literatura C-MAPSS porque o desgaste físico não se manifesta nos sensores antes dos últimos ~125 ciclos.
- **Função assimétrica da NASA:** penaliza previsões otimistas/atrasadas ($\hat{y} > y$, divisor 10) mais que previsões conservadoras/adiantadas ($\hat{y} < y$, divisor 13), refletindo o maior custo de uma quebra inesperada na manutenção industrial.
"""
    )

    st.subheader("V1 — AI4I 2020 Predictive Maintenance Dataset")
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
- **Dados de simulação:** tanto o AI4I 2020 quanto o NASA C-MAPSS vêm de simuladores validados na literatura; não substituem a calibração com o histórico real da própria planta.
- **No AI4I 2020 (V1):** cada linha é um instante isolado e a falha por desgaste da ferramenta (TWF) ocorre num momento sorteado entre ~200 e 240 min de uso.
- **No C-MAPSS FD001 (V2):** há uma única condição operacional (nível do mar) e um único modo de degradação (compressor de alta pressão — HPC). Para múltiplas condições operacionais e múltiplos modos de falha, usam-se os subconjuntos FD002 a FD004.
"""
    )
    st.subheader("Fontes e licença")
    st.markdown(
        """
- **NASA C-MAPSS:** A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08, 2008.
- **AI4I 2020:** UCI Machine Learning Repository — [https://doi.org/10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C) (CC BY 4.0).
- **Código do projeto:** licença MIT — [repositório no GitHub](https://github.com/DenisPaulo/manutencao-preditiva-ia).
"""
    )
