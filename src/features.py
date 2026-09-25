"""Preparação dos dados para o modelo de previsão de falhas.

Uso pela linha de comando (a partir da raiz do projeto):

    python -m src.features

Gera em ``data/processed/`` os arquivos ``X_train.csv``, ``X_test.csv``,
``y_train.csv`` e ``y_test.csv``. A etapa de treino pode reaproveitar tudo
chamando :func:`get_train_test`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import PROJECT_ROOT, load_data

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Colunas de identificação: não carregam informação física sobre a máquina
ID_COLUMNS = ["UDI", "Product ID"]

# Modos de falha (TWF, HDF, PWF, OSF, RNF).
# ATENÇÃO — VAZAMENTO DE ALVO (data leakage): essas colunas dizem *qual* falha
# aconteceu, ou seja, só são conhecidas DEPOIS que a máquina falhou. Se ficarem
# como entrada, o modelo "cola" a resposta (Machine failure = 1 sempre que uma
# delas for 1) e parece perfeito no teste, mas é inútil na fábrica, onde
# queremos prever a falha ANTES dela acontecer. Por isso são removidas.
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

RAW_TARGET = "Machine failure"
TARGET = "machine_failure"

# Tipo/qualidade do produto: L (baixa), M (média), H (alta).
# Existe uma ordem natural, então usamos codificação ordinal (0, 1, 2) em vez de
# one-hot — funciona bem com modelos de árvore como o XGBoost.
TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}


def clean_column_name(name: str) -> str:
    """Converte 'Air temperature [K]' em 'air_temperature_k' (snake_case).

    O XGBoost não aceita colchetes nem sinais de '<' nos nomes das colunas.
    """
    name = name.strip().lower()
    name = re.sub(r"[\[\]\(\)]", " ", name)      # remove colchetes/parênteses
    name = re.sub(r"[^0-9a-z]+", "_", name)      # qualquer outro símbolo vira '_'
    return name.strip("_")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis com significado físico a partir dos sensores.

    - ``power_w``: potência mecânica no eixo, P = torque [Nm] x velocidade
      angular [rad/s], onde rad/s = rpm x 2π / 60.
    - ``temp_diff_k``: diferença entre temperatura de processo e do ar
      (indica quão bem o calor está sendo dissipado).
    """
    df = df.copy()
    df["power_w"] = (df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60).round(2)
    # Arredonda em 0,1 K (mesma resolução dos sensores) para evitar ruído de ponto flutuante
    df["temp_diff_k"] = (df["process_temperature_k"] - df["air_temperature_k"]).round(1)
    return df


def prepare_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Transforma o DataFrame bruto em (X, y) prontos para o modelo."""
    df = df_raw.drop(columns=ID_COLUMNS + FAILURE_MODE_COLUMNS)
    df = df.rename(columns=clean_column_name)

    unknown = set(df["type"].unique()) - set(TYPE_MAPPING)
    if unknown:
        raise ValueError(f"Valores inesperados na coluna Type: {unknown}")
    df["type"] = df["type"].map(TYPE_MAPPING).astype("int64")

    df = add_features(df)

    y = df.pop(TARGET).astype("int64")
    X = df

    # Garantia extra: nenhuma coluna de modo de falha pode sobrar em X
    leaked = {clean_column_name(c) for c in FAILURE_MODE_COLUMNS} & set(X.columns)
    assert not leaked, f"Colunas com vazamento de alvo em X: {leaked}"
    return X, y


def build_feature_row(
    product_type: str,
    air_temperature_k: float,
    process_temperature_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> pd.DataFrame:
    """Monta UMA linha de entrada do modelo a partir das leituras dos sensores.

    Usado pelo app (Streamlit) e pela explicação local: aplica exatamente as
    mesmas transformações do treino (codificação de Type e variáveis físicas).
    """
    if product_type not in TYPE_MAPPING:
        raise ValueError(f"Type deve ser um de {list(TYPE_MAPPING)}, recebido: {product_type!r}")
    row = pd.DataFrame([{
        "type": TYPE_MAPPING[product_type],
        "air_temperature_k": float(air_temperature_k),
        "process_temperature_k": float(process_temperature_k),
        "rotational_speed_rpm": float(rotational_speed_rpm),
        "torque_nm": float(torque_nm),
        "tool_wear_min": float(tool_wear_min),
    }])
    return add_features(row)


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divisão treino/teste 80/20 estratificada pelo alvo.

    A estratificação mantém a mesma proporção de falhas (~3,4%) nos dois
    conjuntos — essencial quando a classe positiva é rara.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def get_train_test(
    df_raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Função principal para reuso no treino: devolve X_train, X_test, y_train, y_test."""
    if df_raw is None:
        df_raw = load_data()
    X, y = prepare_features(df_raw)
    return split_data(X, y)


def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    """Salva os conjuntos em CSV (a pasta data/processed/ fica fora do Git)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_frame().to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_frame().to_csv(out_dir / "y_test.csv", index=False)


def load_processed(
    in_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Lê os conjuntos já salvos em data/processed/."""
    X_train = pd.read_csv(in_dir / "X_train.csv")
    X_test = pd.read_csv(in_dir / "X_test.csv")
    y_train = pd.read_csv(in_dir / "y_train.csv")[TARGET]
    y_test = pd.read_csv(in_dir / "y_test.csv")[TARGET]
    return X_train, X_test, y_train, y_test


def main() -> None:
    X_train, X_test, y_train, y_test = get_train_test()
    save_processed(X_train, X_test, y_train, y_test)

    print(f"[preparação] Features ({X_train.shape[1]}): {list(X_train.columns)}")
    for nome, y in (("treino", y_train), ("teste", y_test)):
        print(
            f"[preparação] {nome}: {len(y)} linhas | falhas: {int(y.sum())} "
            f"({y.mean():.2%})"
        )
    print(f"[preparação] Arquivos salvos em {PROCESSED_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
