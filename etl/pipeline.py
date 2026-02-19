"""
ETL Pipeline - Precificação de Imóveis
Extrai dados simulando Kaggle/APIs públicas, transforma e carrega em DW SQLite
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(__file__))  # project root
DB_PATH = os.path.join(_ROOT, "data", "imoveis_dw.db")
DATA_PATH = os.path.join(_ROOT, "data")


# ─────────────────────────────────────────────
# EXTRACT
# ─────────────────────────────────────────────
def extract_data(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Simula extração de dados de uma API pública / dataset Kaggle de imóveis.
    Em produção: substituir por requests para API ou leitura de CSV real.
    """
    logger.info("Iniciando extração de dados...")
    np.random.seed(seed)

    bairros_config = {
        "Jardins": {
            "base_preco": 12000,
            "std": 3000,
            "renda_media": 18000,
            "idhm": 0.91,
        },
        "Itaim Bibi": {
            "base_preco": 11000,
            "std": 2800,
            "renda_media": 16000,
            "idhm": 0.90,
        },
        "Moema": {"base_preco": 9500, "std": 2500, "renda_media": 14000, "idhm": 0.88},
        "Pinheiros": {
            "base_preco": 8500,
            "std": 2200,
            "renda_media": 12000,
            "idhm": 0.87,
        },
        "Vila Madalena": {
            "base_preco": 8000,
            "std": 2000,
            "renda_media": 11000,
            "idhm": 0.86,
        },
        "Consolação": {
            "base_preco": 7500,
            "std": 1800,
            "renda_media": 10000,
            "idhm": 0.85,
        },
        "Santana": {"base_preco": 6000, "std": 1500, "renda_media": 8000, "idhm": 0.82},
        "Tatuapé": {"base_preco": 5800, "std": 1400, "renda_media": 7500, "idhm": 0.81},
        "Vila Mariana": {
            "base_preco": 7000,
            "std": 1700,
            "renda_media": 9500,
            "idhm": 0.84,
        },
        "Lapa": {"base_preco": 5500, "std": 1300, "renda_media": 7000, "idhm": 0.80},
        "Penha": {"base_preco": 4200, "std": 1100, "renda_media": 5500, "idhm": 0.76},
        "Jabaquara": {
            "base_preco": 4500,
            "std": 1200,
            "renda_media": 5800,
            "idhm": 0.77,
        },
        "Guaianazes": {
            "base_preco": 3200,
            "std": 900,
            "renda_media": 4000,
            "idhm": 0.72,
        },
        "Cidade Tiradentes": {
            "base_preco": 2800,
            "std": 800,
            "renda_media": 3200,
            "idhm": 0.70,
        },
        "Capão Redondo": {
            "base_preco": 3000,
            "std": 850,
            "renda_media": 3500,
            "idhm": 0.71,
        },
    }

    tipos = ["Apartamento", "Casa", "Studio", "Cobertura", "Kitnet"]
    tipo_multiplicador = {
        "Apartamento": 1.0,
        "Casa": 1.2,
        "Studio": 0.8,
        "Cobertura": 1.6,
        "Kitnet": 0.7,
    }

    registros = []
    bairros_list = list(bairros_config.keys())
    pesos = [
        0.10,
        0.09,
        0.08,
        0.08,
        0.07,
        0.06,
        0.07,
        0.07,
        0.07,
        0.06,
        0.06,
        0.05,
        0.05,
        0.04,
        0.05,
    ]

    for i in range(n_samples):
        bairro = np.random.choice(bairros_list, p=pesos)
        cfg = bairros_config[bairro]
        tipo = np.random.choice(tipos, p=[0.50, 0.20, 0.12, 0.08, 0.10])

        quartos = np.random.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10])
        banheiros = max(1, quartos - np.random.randint(0, 2))
        vagas = np.random.choice([0, 1, 2, 3], p=[0.15, 0.45, 0.30, 0.10])

        area_base = {1: 45, 2: 70, 3: 100, 4: 140}[quartos]
        area = max(20, np.random.normal(area_base, area_base * 0.20))
        area = round(area, 1)

        condominio = round(np.random.uniform(300, 2500), 0) if tipo != "Casa" else 0
        andar = (
            np.random.randint(1, 25)
            if tipo in ["Apartamento", "Cobertura", "Studio", "Kitnet"]
            else 0
        )
        ano_construcao = np.random.randint(1970, 2024)
        idade = 2024 - ano_construcao

        # Preço por m²
        preco_m2 = max(1000, np.random.normal(cfg["base_preco"], cfg["std"]))
        preco_m2 *= tipo_multiplicador[tipo]
        preco_m2 *= 1 + quartos * 0.03
        preco_m2 *= 1 + vagas * 0.04
        preco_m2 *= max(0.7, 1 - idade * 0.003)
        if andar > 10:
            preco_m2 *= 1.05

        preco_total = round(preco_m2 * area, -3)
        preco_m2 = round(preco_m2, 0)

        registros.append(
            {
                "id": i + 1,
                "bairro": bairro,
                "tipo": tipo,
                "area_m2": area,
                "quartos": quartos,
                "banheiros": banheiros,
                "vagas": vagas,
                "andar": andar,
                "ano_construcao": ano_construcao,
                "idade_imovel": idade,
                "condominio": condominio,
                "preco_m2": preco_m2,
                "preco_venda": preco_total,
                "renda_media_bairro": cfg["renda_media"],
                "idhm_bairro": cfg["idhm"],
                "latitude": -23.55 + np.random.normal(0, 0.05),
                "longitude": -46.63 + np.random.normal(0, 0.05),
                "data_extracao": datetime.now().strftime("%Y-%m-%d"),
                "fonte": "Kaggle Dataset SP",
            }
        )

    df = pd.DataFrame(registros)
    logger.info(f"Extração concluída: {len(df)} registros")
    return df


# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
def transform_data(df: pd.DataFrame) -> dict:
    """Normaliza, cria features e prepara tabelas dimensionais."""
    logger.info("Iniciando transformação...")

    df = df.copy()
    df.dropna(inplace=True)

    # Normalização z-score de variáveis numéricas
    cols_normalizar = ["area_m2", "preco_m2"]
    stats = {}
    for col in cols_normalizar:
        media = df[col].mean()
        std = df[col].std()
        df[f"{col}_norm"] = (df[col] - media) / std
        stats[col] = {"mean": media, "std": std}

    # Log do preço (melhora linearidade)
    df["log_preco"] = np.log1p(df["preco_venda"])
    df["log_area"] = np.log1p(df["area_m2"])

    # Feature: custo por andar
    df["preco_por_quarto"] = df["preco_venda"] / df["quartos"]

    # Tabela Dimensão Bairro
    dim_bairro = (
        df.groupby("bairro")
        .agg(
            preco_medio=("preco_venda", "mean"),
            preco_m2_medio=("preco_m2", "mean"),
            renda_media=("renda_media_bairro", "first"),
            idhm=("idhm_bairro", "first"),
            total_imoveis=("id", "count"),
            lat_media=("latitude", "mean"),
            lon_media=("longitude", "mean"),
        )
        .reset_index()
    )

    # Tabela Dimensão Tipo
    dim_tipo = (
        df.groupby("tipo")
        .agg(
            preco_medio=("preco_venda", "mean"),
            area_media=("area_m2", "mean"),
            total=("id", "count"),
        )
        .reset_index()
    )

    # Fato principal
    fato_imoveis = df[
        [
            "id",
            "bairro",
            "tipo",
            "area_m2",
            "quartos",
            "banheiros",
            "vagas",
            "andar",
            "idade_imovel",
            "condominio",
            "preco_m2",
            "preco_venda",
            "log_preco",
            "log_area",
            "preco_por_quarto",
            "area_m2_norm",
            "preco_m2_norm",
            "renda_media_bairro",
            "idhm_bairro",
            "latitude",
            "longitude",
            "data_extracao",
            "fonte",
        ]
    ]

    logger.info("Transformação concluída.")
    return {
        "fato_imoveis": fato_imoveis,
        "dim_bairro": dim_bairro,
        "dim_tipo": dim_tipo,
        "stats": stats,
    }


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_data(transformed: dict) -> None:
    """Carrega dados no Data Warehouse SQLite."""
    logger.info("Iniciando carga no DW...")
    os.makedirs(DATA_PATH, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    transformed["fato_imoveis"].to_sql(
        "fato_imoveis", conn, if_exists="replace", index=False
    )
    transformed["dim_bairro"].to_sql(
        "dim_bairro", conn, if_exists="replace", index=False
    )
    transformed["dim_tipo"].to_sql("dim_tipo", conn, if_exists="replace", index=False)

    # Salva stats de normalização
    stats_df = pd.DataFrame(transformed["stats"]).T.reset_index()
    stats_df.columns = ["coluna", "mean", "std"]
    stats_df.to_sql("normalizacao_stats", conn, if_exists="replace", index=False)

    conn.close()
    logger.info(f"DW atualizado em: {DB_PATH}")


# ─────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────
def run_pipeline(n_samples: int = 1500) -> dict:
    raw = extract_data(n_samples=n_samples)
    transformed = transform_data(raw)
    load_data(transformed)
    logger.info("Pipeline ETL finalizado com sucesso!")
    return transformed


if __name__ == "__main__":
    run_pipeline()
