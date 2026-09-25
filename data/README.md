# Dados — AI4I 2020 Predictive Maintenance Dataset

Este projeto usa o **AI4I 2020 Predictive Maintenance Dataset**, publicado no
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(id 601). É um conjunto **sintético**, criado por Stephan Matzka para reproduzir o
comportamento de dados reais de manutenção preditiva de uma máquina de usinagem:
10.000 registros de operação, cada um com leituras de sensores e a indicação de
falha da máquina.

> O CSV **não é versionado** neste repositório (a pasta `data/raw/` está no
> `.gitignore`). Ele é baixado direto da fonte oficial.

## Como baixar

A partir da raiz do projeto, com o ambiente Python ativo:

```bash
pip install -r requirements.txt
python -m src.data        # baixa e extrai data/raw/ai4i2020.csv (só na primeira vez)
python -m src.features    # gera os conjuntos de treino/teste em data/processed/
```

O script baixa o arquivo oficial:

```
https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip
```

e extrai o `ai4i2020.csv`. Se o arquivo já existir, o download é ignorado.

Download manual (alternativa): baixe o `.zip` pelo link acima e coloque o
`ai4i2020.csv` em `data/raw/`.

## Estrutura das pastas

| Pasta             | Conteúdo                                                      | Versionada? |
|-------------------|---------------------------------------------------------------|-------------|
| `data/raw/`       | CSV original, exatamente como veio do UCI                     | Não         |
| `data/processed/` | `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`      | Não         |

## Colunas explicadas (visão de chão de fábrica)

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
| `Machine failure` | 0/1 | **Alvo do modelo**: 1 se a máquina falhou naquele registro, por qualquer um dos modos abaixo. |
| `TWF` | 0/1 | *Tool Wear Failure* — falha por desgaste da ferramenta (fim de vida útil). |
| `HDF` | 0/1 | *Heat Dissipation Failure* — falha por dissipação de calor insuficiente (pouca diferença entre ar e processo com rotação baixa). |
| `PWF` | 0/1 | *Power Failure* — potência (torque × rotação) fora da faixa de trabalho: muito baixa ou muito alta. |
| `OSF` | 0/1 | *Overstrain Failure* — sobrecarga: combinação de ferramenta gasta com torque alto. |
| `RNF` | 0/1 | *Random Failure* — falha aleatória, sem relação com os parâmetros de processo. |

**Importante:** as colunas `TWF`, `HDF`, `PWF`, `OSF` e `RNF` descrevem *qual*
falha aconteceu — só são conhecidas depois da quebra. Por isso elas **não entram
como variáveis de entrada** do modelo (seria vazamento de alvo / *data leakage*).
Veja `src/features.py` e `notebooks/02_preparacao.ipynb`.

A descrição oficial no UCI traz a regra de geração de cada modo de falha. Os
números citados nesse texto (por exemplo, 120 casos de TWF e 5 de RNF) **não
batem** com a contagem feita no próprio CSV (46 de TWF e 19 de RNF, por
exemplo). Neste projeto valem sempre os números calculados a partir do arquivo —
veja `notebooks/01_exploracao.ipynb`.

## Licença e citação

- **Fonte:** UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset (id 601)
- **Autor:** Stephan Matzka
- **Licença:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **DOI:** [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)

Citação do dataset:

> AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

Artigo introdutório:

> S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," *2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, 2020, pp. 69–74. doi: 10.1109/AI4I49448.2020.00023
