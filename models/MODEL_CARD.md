# Model Card — Manutenção Preditiva com IA (V1 AI4I 2020 & V2 NASA C-MAPSS RUL)

## O que os modelos fazem
- **Versão 1 — Classificador de falha (`models/xgb_model.json`, AI4I 2020):** recebe leituras instantâneas de sensores de uma máquina de usinagem e devolve a **probabilidade de falha** e a explicação SHAP dessa previsão.
- **Versão 2 — Regressor de Vida Útil Remanescente / RUL (`models/xgb_rul_fd001.json`, NASA C-MAPSS FD001):** recebe a série temporal de sensores de uma turbina (média móvel, desvio-padrão e variação frente ao início de operação do motor) e prevê **quantos ciclos restam até a falha** (limitado em 125 ciclos), classificando o equipamento em **Crítico (≤ 20 ciclos)**, **Atenção (21–50 ciclos)** ou **Saudável (> 50 ciclos)** com explicação SHAP em ciclos.

## Uso pretendido
- Portfólio e estudo de manutenção preditiva explicável aplicada à manutenção industrial.
- **Não** usar para decidir intervenção em máquina real sem validação e calibração com dados da própria planta.

---

## Versão 2 — Regressor de Vida Útil Remanescente (NASA C-MAPSS FD001)

### Dados e preparação (`src/cmapss.py`)
- **Conjunto:** NASA C-MAPSS subconjunto `FD001` (1 condição operacional ao nível do mar, 1 modo de degradação no compressor de alta pressão — HPC).
- **Divisão oficial:** treino com **100 motores (20.631 ciclos)** operando até a falha; teste independente com **100 motores (13.096 ciclos)** truncados antes da falha e avaliados no último ciclo observado contra `RUL_FD001.txt`.
- **Limpeza de sensores:** removidos `op_setting_3` e os 7 sensores constantes no FD001 (`s1, s5, s6, s10, s16, s18, s19`); mantidos os **14 sensores informativos** (`s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`).
- **Engenharia de variáveis (57 features):** ciclo atual + para cada um dos 14 sensores: leitura atual, média móvel de 5 ciclos (`_mean5`), desvio-padrão móvel de 5 ciclos (`_std5`) e diferença entre a média móvel atual e a média dos 5 primeiros ciclos do próprio motor (`_delta`).
- **Teto linear (*piecewise linear RUL*):** alvo limitado em `RUL_CAP = 125` ciclos no treino e no teste, pois motores saudáveis não apresentam assinatura de degradação antes dos últimos ~125 ciclos.

### Modelos e métricas no conjunto de teste oficial (100 motores)

| Modelo | RMSE [ciclos] ↓ | MAE [ciclos] ↓ | R² ↑ | NASA Score ↓ | NASA Score / motor ↓ |
|---|---|---|---|---|---|
| Base (`Ridge`, $\alpha=10$) | 16,60 | 13,23 | 0,8285 | 445,1 | 4,45 |
| **Principal (`XGBRegressor`)** | **11,13** | **8,31** | **0,9228** | **176,8** | **1,77** |

- **Por que usar a Pontuação Assimétrica da NASA (PHM08)?** Para um erro $d = \hat{y} - y$, a penalidade é $e^{-d/13} - 1$ quando $d < 0$ (previsão conservadora/adiantada) e $e^{d/10} - 1$ quando $d \ge 0$ (previsão otimista/atrasada). Na manutenção industrial, superestimar a vida útil restante leva à quebra inesperada da máquina em produção.

### Explicabilidade (SHAP em ciclos)
- Os valores SHAP do `XGBRegressor` estão diretamente em **ciclos de vida útil**: as variáveis que mais deslocam a previsão no teste são `s11_delta` (subida da pressão estática no HPC, média de |SHAP| = 10,70 ciclos), `s4_delta` (subida da temperatura de saída da LPT, 5,99 ciclos), `cycle` (5,27 ciclos), `s13_delta` (3,70 ciclos) e `s14_delta` (3,43 ciclos).

---

## Versão 1 — Classificador de falha (AI4I 2020)

### Dados
- AI4I 2020 (UCI, CC BY 4.0), 10.000 registros sintéticos, cerca de 3,4% de falhas.
- Removidos: `UDI`, `Product ID` e as colunas de modo de falha (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`), que vazariam a resposta.
- Criados: `power_w` (potência = torque × rotação em rad/s) e `temp_diff_k` (temperatura do processo − temperatura do ar).
- Divisão treino/teste: 80/20, estratificada pelo alvo `Machine failure`, `random_state=42`. Treino: 8.000 registros (271 falhas); teste: 2.000 registros (68 falhas). O teste só foi usado na avaliação final.

### Modelos
| Papel | Algoritmo | Tratamento do desbalanceamento |
|---|---|---|
| Base | Regressão logística | Dados padronizados e `class_weight='balanced'` |
| Principal | XGBoost | `scale_pos_weight` = 28,52 (7.729 sem falha ÷ 271 falhas no treino); hiperparâmetros por busca aleatória (20 combinações, validação cruzada estratificada de 5 dobras no treino, métrica PR-AUC) |

### Métricas (conjunto de teste)
<!-- Somente números da execução real. -->
| Modelo | Recall | Precisão | F1 | PR-AUC |
|---|---|---|---|---|
| Regressão logística | 0,765 | 0,308 | 0,439 | 0,469 |
| XGBoost | 0,824 | 0,757 | 0,789 | 0,884 |

Cada modelo no seu limiar escolhido (XGBoost 0,4969; regressão logística 0,7349). Para comparação, um modelo que nunca prevê falha teria acurácia de 96,6% e recall 0.

Matriz de confusão do XGBoost no limiar escolhido: 56 falhas detectadas, 12 falhas não detectadas, 18 alarmes falsos e 1.914 acertos sem falha (de 1.932 registros sem falha). IC 95% por bootstrap do recall: 0,727 a 0,908.

### Ponto de corte e explicabilidade
- Limiar escolhido: 0,4969 (maximiza F2 nas previsões *out-of-fold* da validação cruzada no treino).
- Importância global (SHAP, média de |valor SHAP| no teste): desgaste da ferramenta, rotação, potência, torque e diferença de temperatura.

## Limitações e riscos
- Dados de simulação; desempenho em equipamento real depende de instrumentação e calibração na própria planta.
- Na V1, não há componente temporal e a falha aleatória (`RNF`) não é previsível por definição.
- Na V2 (FD001), todas as turbinas operam na mesma condição (nível do mar) e sofrem degradação no mesmo subconjunto (HPC).

## Versão
- v0.1 — 25/09/2026 (Classificador AI4I 2020)
- v0.2 — 24/10/2026 (Regressor de RUL NASA C-MAPSS FD001)
