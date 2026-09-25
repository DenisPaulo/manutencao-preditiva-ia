# 🔧 Manutenção Preditiva com IA

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-modelo-EB5E28)
![SHAP](https://img.shields.io/badge/SHAP-explicabilidade-8A2BE2)
![Streamlit](https://img.shields.io/badge/Streamlit-painel-FF4B4B?logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> **Com essas leituras dos sensores, a máquina vai falhar? E por quê?**
> Modelo de classificação de falhas com explicação de cada previsão, construído por quem vem do chão de fábrica (robôs ABB, CLP e manutenção preventiva).

🔗 **Demo:** _link do Streamlit Cloud (em breve)_

<!-- GIF/print do painel aqui quando o passo 8 estiver pronto: docs/painel.gif -->

---

## O problema

Em manutenção, parar a máquina tarde demais custa uma quebra; parar cedo demais custa produção. Este projeto usa leituras de sensores para estimar a **probabilidade de falha** e mostra **quais sinais puxaram o risco para cima**, para que a decisão de intervir seja de quem conhece a máquina, e não de uma caixa-preta.

## Os dados

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository), um conjunto sintético que reproduz dados reais de manutenção.

- 10.000 registros, com falha em cerca de 3,4% deles (desbalanceado, como numa fábrica).
- Variáveis: tipo de produto (L, M, H), temperatura do ar, temperatura do processo, rotação, torque e desgaste da ferramenta.
- O CSV **não** é versionado. Veja [`data/README.md`](data/README.md) para baixar.

## Como funciona

| Etapa | O que acontece | Onde |
|---|---|---|
| 1. Exploração | Gráficos de cada sensor em máquinas que falharam e que não falharam | `notebooks/01_exploracao.ipynb` |
| 2. Preparação | Remove IDs e colunas de tipo de falha (evita vazamento da resposta); cria **potência** (torque × rotação) e **ΔT** (processo − ar) | `src/` |
| 3. Modelo | Regressão logística como base de comparação e **XGBoost** como modelo principal, com peso maior para a classe de falha | `src/` |
| 4. Avaliação | Recall, precisão, matriz de confusão e escolha do ponto de corte do alarme | `notebooks/`, `models/MODEL_CARD.md` |
| 5. Explicação | **SHAP** global (quais sensores pesam mais) e local (por que *esta* previsão) | `src/`, painel |
| 6. Painel | Controles de sensores com risco em tempo real e gráfico do porquê; abas de métricas e dados | `app/streamlit_app.py` |

## Resultado

<!-- Preencher SOMENTE com números da execução real (conjunto de teste). -->

| Modelo | Recall (falha) | Precisão (falha) | F1 | PR-AUC |
|---|---|---|---|---|
| Regressão logística | 0,765 | 0,308 | 0,439 | 0,469 |
| XGBoost | 0,824 | 0,757 | 0,789 | 0,884 |

Conjunto de teste: 2.000 registros, 68 falhas. Cada modelo no seu limiar escolhido no treino (XGBoost 0,4969; regressão logística 0,7349). O XGBoost detecta 56 das 68 falhas com 18 alarmes falsos.

**Por que não acurácia?** Um modelo que responde "nunca falha" acerta cerca de 97% e não pega nenhuma falha. Por isso o foco é em quantas falhas reais o modelo detecta (recall) e quantos alarmes falsos ele gera (precisão).

## Decisões

- **AI4I antes do C-MAPSS:** a pergunta "vai falhar, sim ou não?" é mais fácil de explicar e validar numa primeira versão.
- **Colunas de modo de falha removidas:** elas contêm a resposta e deixariam o resultado artificialmente perfeito.
- **Ponto de corte como decisão de manutenção:** baixar o limiar pega mais falhas, mas para a máquina à toa com mais frequência. O valor escolhido e o motivo estão no [model card](models/MODEL_CARD.md).
- **Variáveis de chão de fábrica:** potência e ΔT vêm da física do processo, não de tentativa e erro.

## Limitações

- Os dados são **sintéticos**: o modelo demonstra o método, não substitui validação em máquina real.
- Não considera histórico temporal (cada linha é um instante isolado).
- 9 das 12 falhas não detectadas no teste são por desgaste da ferramenta (TWF), que neste dataset acontece num momento sorteado entre ~200 e 240 min de uso (no CSV, de 198 a 253 min). Os sensores não dão aviso antecipado.
- Métricas valem para a distribuição deste dataset.

## Como rodar

```bash
git clone https://github.com/DenisPaulo/manutencao-preditiva-ia.git
cd manutencao-preditiva-ia
pip install -r requirements.txt
# baixar os dados: ver data/README.md
streamlit run app/streamlit_app.py
```

## Estrutura

```
├── data/                  # instruções de download (CSV fora do Git)
├── notebooks/             # exploração e avaliação
├── src/                   # preparação, treino e explicação
├── app/streamlit_app.py   # painel
├── models/MODEL_CARD.md   # o que o modelo faz, métricas e limites
└── requirements.txt
```

## Próximo

- Versão 2 com **NASA C-MAPSS**: prever quanto tempo falta até a falha (vida útil remanescente).
- Alerta simples quando o risco passa do limiar.

## Fonte e licença

- Dados: Matzka, S. *Explainable Artificial Intelligence for Predictive Maintenance Applications*, AI4I 2020. Disponível no [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), licença CC BY 4.0.
- Código: [MIT](LICENSE).

---

Feito por [Denis Paulo](https://github.com/DenisPaulo) · da automação industrial para IA & Dados.
