# Dados — AI4I 2020 (V1) e NASA C-MAPSS FD001 (V2)

Este projeto utiliza dois conjuntos públicos de referência em manutenção preditiva:

1. **V1 — AI4I 2020 Predictive Maintenance Dataset** ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), id 601): 10.000 registros de operação de uma máquina de usinagem para **classificação de falha** (vai falhar: sim/não).
2. **V2 — NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (FD001)** (NASA Prognostics Data Repository, PHM08): séries temporais ciclo a ciclo de 100 motores de treino e 100 motores de teste para **regressão de Vida Útil Remanescente (RUL)**.

> Os arquivos brutos **não são versionados** neste repositório (a pasta `data/raw/` está no `.gitignore`). Os scripts baixam os arquivos automaticamente quando executados.

---

## Como baixar

A partir da raiz do projeto, com o ambiente Python ativo:

```bash
pip install -r requirements.txt

# V1 — AI4I 2020 (classificação)
python -m src.data        # baixa e extrai data/raw/ai4i2020.csv
python -m src.features    # gera os conjuntos de treino/teste em data/processed/

# V2 — NASA C-MAPSS FD001 (vida útil remanescente / RUL)
python -m src.cmapss      # baixa data/raw/cmapss/*FD001.txt, treina o modelo e atualiza models/
```

Se o ambiente não tiver acesso à internet no momento da execução dos testes, `src/cmapss.py` possui um gerador sintético determinístico (`make_synthetic_fd001`) usado automaticamente como fallback offline.

## Estrutura das pastas

| Pasta | Conteúdo | Versionada? |
|---|---|---|
| `data/raw/` | `ai4i2020.csv` (UCI) | Não |
| `data/raw/cmapss/` | `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` (NASA C-MAPSS) | Não |
| `data/processed/` | `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` (V1) | Não |

---

## V2 — NASA C-MAPSS FD001 explicado (visão de manutenção industrial)

No subconjunto **FD001**, cada linha representa **1 ciclo completo de operação** de uma turbina a gás (condição ao nível do mar, modo de falha por degradação do compressor de alta pressão — HPC):

- **`train_FD001.txt` (20.631 linhas, 100 motores):** cada motor opera desde a condição saudável até a falha funcional (o último ciclo do motor tem `RUL = 0`).
- **`test_FD001.txt` (13.096 linhas, 100 motores):** as leituras são cortadas antes da falha; o objetivo é prever quantos ciclos restam após o último ciclo observado de cada motor.
- **`RUL_FD001.txt` (100 linhas):** a vida útil remanescente real (em ciclos) no instante do corte de cada um dos 100 motores de teste.

### Sensores mantidos vs. sensores removidos no FD001

Das 26 colunas originais (`unit_id`, `cycle`, 3 ajustes operacionais e 21 sensores `s1..s21`), **8 colunas são constantes/planas** no FD001 (`op_setting_3`, `s1`, `s5`, `s6`, `s10`, `s16`, `s18`, `s19`) e são descartadas por não trazerem informação sobre o desgaste. Os **14 sensores informativos** mantidos são:

| Sensor | Grandeza física | Tendência com a degradação |
|---|---|---|
| `s2` | Temperatura na saída do compressor de baixa pressão (LPC) [°R] | Sobe |
| `s3` | Temperatura na saída do compressor de alta pressão (HPC) [°R] | Sobe |
| `s4` | Temperatura na saída da turbina de baixa pressão (LPT) [°R] | Sobe |
| `s7` | Pressão total na saída do HPC [psia] | Cai |
| `s8` | Rotação física do fan [rpm] | Sobe |
| `s9` | Rotação física do núcleo (core) [rpm] | Sobe |
| `s11` | Pressão estática na saída do HPC [psia] | Sobe (principal indicador no SHAP) |
| `s12` | Razão entre fluxo de combustível e pressão estática Ps30 [pps/psi] | Cai |
| `s13` | Rotação corrigida do fan [rpm] | Sobe |
| `s14` | Rotação corrigida do núcleo [rpm] | Sobe |
| `s15` | Razão de bypass (BPR) | Sobe |
| `s17` | Entalpia de sangria (*bleed enthalpy*) | Sobe |
| `s20` | Fluxo de ar de resfriamento da turbina de alta pressão (HPT) [lbm/s] | Cai |
| `s21` | Fluxo de ar de resfriamento da turbina de baixa pressão (LPT) [lbm/s] | Cai |

---

## V1 — Colunas do AI4I 2020 explicadas (visão de manutenção industrial)

| Coluna | Unidade | O que significa na prática |
|---|---|---|
| `UDI` | — | Número sequencial do registro (1 a 10.000). Só um identificador. |
| `Product ID` | — | Código da peça produzida: letra da variante (L/M/H) + número de série. Identificador. |
| `Type` | — | Qualidade/variante do produto: **L** = baixa (60% dos registros), **M** = média (30%), **H** = alta (10%). Produtos de maior qualidade exigem mais da ferramenta. |
| `Air temperature [K]` | Kelvin | Temperatura ambiente perto da máquina (~300 K ≈ 27 °C). |
| `Process temperature [K]` | Kelvin | Temperatura na região de trabalho/processo (~310 K ≈ 37 °C). Acompanha a temperatura ambiente, cerca de 10 K acima. |
| `Rotational speed [rpm]` | rpm | Velocidade do eixo/fuso da máquina. |
| `Torque [Nm]` | N·m | Esforço de giro no eixo — quanto "força" a máquina está fazendo no corte. |
| `Tool wear [min]` | minutos | Tempo de uso acumulado da ferramenta de corte desde a última troca. É o "horímetro" da ferramenta. |
| `Machine failure` | 0/1 | **Alvo do modelo V1**: 1 se a máquina falhou naquele registro, por qualquer um dos modos abaixo. |
| `TWF` | 0/1 | *Tool Wear Failure* — falha por desgaste da ferramenta (fim de vida útil). |
| `HDF` | 0/1 | *Heat Dissipation Failure* — falha por dissipação de calor insuficiente (pouca diferença entre ar e processo com rotação baixa). |
| `PWF` | 0/1 | *Power Failure* — potência (torque × rotação) fora da faixa de trabalho: muito baixa ou muito alta. |
| `OSF` | 0/1 | *Overstrain Failure* — sobrecarga: combinação de ferramenta gasta com torque alto. |
| `RNF` | 0/1 | *Random Failure* — falha aleatória, sem relação com os parâmetros de processo. |

**Importante:** as colunas `TWF`, `HDF`, `PWF`, `OSF` e `RNF` descrevem *qual*
falha aconteceu — só são conhecidas depois da quebra. Por isso elas **não entram
como variáveis de entrada** do modelo (seria vazamento de alvo / *data leakage*).
Veja `src/features.py` e `notebooks/02_preparacao.ipynb`.

## Licença e citação

- **V1 (AI4I 2020):** Stephan Matzka, UCI Machine Learning Repository (id 601), licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
- **V2 (NASA C-MAPSS):** A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", *Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08)*, Denver, CO, 2008.
