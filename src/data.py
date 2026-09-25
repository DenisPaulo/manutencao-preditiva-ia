"""Download e carga do dataset AI4I 2020 Predictive Maintenance (UCI, id 601).

Uso pela linha de comando (a partir da raiz do projeto):

    python -m src.data

O download é idempotente: se o CSV já estiver em ``data/raw/``, nada é baixado
de novo.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# Raiz do projeto (pasta acima de src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CSV_NAME = "ai4i2020.csv"
CSV_PATH = RAW_DIR / CSV_NAME

# Endereço oficial do arquivo no UCI Machine Learning Repository
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

# Quantidade de linhas esperada, usada como verificação simples de integridade
EXPECTED_ROWS = 10_000


def download_dataset(force: bool = False, timeout: int = 60) -> Path:
    """Baixa o .zip do UCI e extrai o ``ai4i2020.csv`` em ``data/raw/``.

    Parâmetros
    ----------
    force:
        Se ``True``, baixa novamente mesmo que o arquivo já exista.
    timeout:
        Tempo máximo (segundos) de espera pela resposta do servidor.

    Retorna
    -------
    Path
        Caminho do CSV extraído.
    """
    if CSV_PATH.exists() and not force:
        print(f"[dados] CSV já existe em {CSV_PATH.relative_to(PROJECT_ROOT)} — download ignorado.")
        return CSV_PATH

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[dados] Baixando {DATASET_URL} ...")
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "manutencao-preditiva-ia"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        if CSV_NAME not in zf.namelist():
            raise FileNotFoundError(
                f"'{CSV_NAME}' não encontrado dentro do zip. Conteúdo: {zf.namelist()}"
            )
        # Grava em arquivo temporário e renomeia: evita CSV pela metade se algo falhar
        tmp_path = CSV_PATH.with_suffix(".csv.tmp")
        tmp_path.write_bytes(zf.read(CSV_NAME))
        tmp_path.replace(CSV_PATH)

    print(f"[dados] CSV salvo em {CSV_PATH.relative_to(PROJECT_ROOT)}")
    return CSV_PATH


def load_data(download: bool = True) -> pd.DataFrame:
    """Carrega o dataset bruto como DataFrame (baixando antes, se necessário)."""
    if download:
        download_dataset()
    elif not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} não existe. Rode `python -m src.data` para baixar o dataset."
        )

    df = pd.read_csv(CSV_PATH)
    if len(df) != EXPECTED_ROWS:
        print(f"[dados] Atenção: esperado {EXPECTED_ROWS} linhas, encontrado {len(df)}.")
    return df


def main() -> None:
    df = load_data()
    n_falhas = int(df["Machine failure"].sum())
    print(f"[dados] {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"[dados] Falhas de máquina: {n_falhas} ({n_falhas / len(df):.2%})")


if __name__ == "__main__":
    main()
