# Model Card — Classificador de falha (AI4I 2020)

## O que o modelo faz
Recebe leituras de sensores de uma máquina (tipo de produto, temperatura do ar, temperatura do processo, rotação, torque, desgaste da ferramenta) e devolve a **probabilidade de falha** e a explicação SHAP dessa previsão.

## Uso pretendido
- Portfólio e estudo de manutenção preditiva explicável.
- **Não** usar para decidir intervenção em máquina real sem validação com dados da própria planta.

## Dados
- AI4I 2020 (UCI, CC BY 4.0), 10.000 registros sintéticos, cerca de 3,4% de falhas.
- Removidos: `UDI`, `Product ID` e as colunas de modo de falha (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`), que vazariam a resposta.
- Criados: `power_w` (potência = torque × rotação em rad/s) e `temp_diff_k` (temperatura do processo − temperatura do ar).
- Divisão treino/teste: 80/20, estratificada pelo alvo `Machine failure`, `random_state=42`. Treino: 8.000 registros (271 falhas); teste: 2.000 registros (68 falhas). O teste só foi usado na avaliação final.

## Modelos
| Papel | Algoritmo | Tratamento do desbalanceamento |
|---|---|---|
| Base | Regressão logística | Dados padronizados e `class_weight='balanced'` |
| Principal | XGBoost | `scale_pos_weight` = 28,52 (7.729 sem falha ÷ 271 falhas no treino); hiperparâmetros por busca aleatória (20 combinações, validação cruzada estratificada de 5 dobras no treino, métrica PR-AUC) |

## Métricas (conjunto de teste)
<!-- Somente números da execução real. -->
| Modelo | Recall | Precisão | F1 | PR-AUC |
|---|---|---|---|---|
| Regressão logística | 0,765 | 0,308 | 0,439 | 0,469 |
| XGBoost | 0,824 | 0,757 | 0,789 | 0,884 |

Cada modelo no seu limiar escolhido (XGBoost 0,4969; regressão logística 0,7349). Para comparação, um modelo que nunca prevê falha teria acurácia de 96,6% e recall 0.

Matriz de confusão do XGBoost no limiar escolhido: 56 falhas detectadas, 12 falhas não detectadas, 18 alarmes falsos e 1.914 acertos sem falha (de 1.932 registros sem falha). IC 95% por bootstrap do recall: 0,727 a 0,908.

## Ponto de corte
- Limiar escolhido: 0,4969 (probabilidade de falha a partir da qual o alarme dispara).
- Motivo: uma falha não detectada custa mais que uma parada desnecessária, então o limiar maximiza o F2 (recall pesa mais que precisão). Foi calculado nas previsões *out-of-fold* da validação cruzada no treino, nunca no teste. O valor ficou próximo de 0,5, e no teste os resultados nos dois limiares são iguais.

## Explicabilidade
- Importância global (SHAP, média de |valor SHAP| no teste): desgaste da ferramenta, rotação, potência, torque e diferença de temperatura; temperaturas do ar e do processo e o tipo de produto pesam pouco.
- Cada previsão no painel mostra as contribuições de cada sensor.

## Limitações e riscos
- Dados sintéticos; desempenho em máquina real é desconhecido.
- Sem componente temporal.
- A falha aleatória (`RNF`) não é previsível por definição.

## Versão
- v0.1 — 25/09/2026
