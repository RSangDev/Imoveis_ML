"""
Testes unitários e de integração para o ETL Pipeline de Precificação de Imóveis
"""

import pytest
import pandas as pd
import sqlite3
from unittest.mock import patch

from etl.pipeline import (
    extract_data,
    transform_data,
    load_data,
    run_pipeline,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def temp_paths(tmp_path):
    """Cria caminhos temporários para evitar poluir o projeto real."""
    db_path = tmp_path / "imoveis_dw_test.db"
    data_path = tmp_path
    return db_path, data_path


@pytest.fixture
def sample_df():
    """DataFrame pequeno e determinístico para os testes."""
    return extract_data(n_samples=100, seed=42)


# ─────────────────────────────────────────────
# TESTS - EXTRACT
# ─────────────────────────────────────────────
def test_extract_data_retorna_dataframe():
    df = extract_data(n_samples=50, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50


def test_extract_data_colunas_esperadas():
    df = extract_data(n_samples=10, seed=42)
    expected_cols = {
        "id",
        "bairro",
        "tipo",
        "area_m2",
        "quartos",
        "banheiros",
        "vagas",
        "andar",
        "ano_construcao",
        "idade_imovel",
        "condominio",
        "preco_m2",
        "preco_venda",
        "renda_media_bairro",
        "idhm_bairro",
        "latitude",
        "longitude",
        "data_extracao",
        "fonte",
    }
    assert expected_cols.issubset(df.columns)


def test_extract_data_deterministico_com_seed():
    df1 = extract_data(100, seed=42)
    df2 = extract_data(100, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_extract_data_precos_sempre_positivos():
    df = extract_data(200, seed=42)
    assert (df["preco_venda"] > 0).all()
    assert (df["preco_m2"] > 1000).all()  # mínimo realista
    assert (df["area_m2"] >= 20).all()


def test_extract_data_varios_tamanhos():
    for n in [0, 10, 50, 500]:
        df = extract_data(n_samples=n, seed=42)
        assert len(df) == n


# ─────────────────────────────────────────────
# TESTS - TRANSFORM
# ─────────────────────────────────────────────
def test_transform_data_estrutura_saida(sample_df):
    result = transform_data(sample_df)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"fato_imoveis", "dim_bairro", "dim_tipo", "stats"}


def test_transform_normalizacao_zscore(sample_df):
    result = transform_data(sample_df)
    fato = result["fato_imoveis"]

    assert "area_m2_norm" in fato.columns
    assert "preco_m2_norm" in fato.columns

    # Média ≈ 0 e desvio padrão ≈ 1
    assert abs(fato["area_m2_norm"].mean()) < 0.01
    assert abs(fato["area_m2_norm"].std() - 1.0) < 0.05
    assert abs(fato["preco_m2_norm"].mean()) < 0.01


def test_transform_features_criadas(sample_df):
    result = transform_data(sample_df)
    fato = result["fato_imoveis"]

    for col in [
        "log_preco",
        "log_area",
        "preco_por_quarto",
        "area_m2_norm",
        "preco_m2_norm",
    ]:
        assert col in fato.columns

    assert (fato["log_preco"] > 0).all()
    assert (fato["preco_por_quarto"] > 0).all()


def test_dim_bairro_agregacao_correta(sample_df):
    result = transform_data(sample_df)
    dim_b = result["dim_bairro"]

    assert len(dim_b) == sample_df["bairro"].nunique()
    assert dim_b["total_imoveis"].sum() == len(sample_df)
    assert "preco_medio" in dim_b.columns
    assert "renda_media" in dim_b.columns


def test_dim_tipo_agregacao_correta(sample_df):
    result = transform_data(sample_df)
    dim_t = result["dim_tipo"]

    assert len(dim_t) == sample_df["tipo"].nunique()
    assert "total" in dim_t.columns
    assert dim_t["total"].sum() == len(sample_df)


# ─────────────────────────────────────────────
# TESTS - LOAD
# ─────────────────────────────────────────────
def test_load_data_cria_tabelas_no_sqlite(temp_paths):
    db_path, data_path = temp_paths
    transformed = transform_data(extract_data(n_samples=30, seed=42))

    with patch("etl.pipeline.DB_PATH", str(db_path)):
        with patch("etl.pipeline.DATA_PATH", str(data_path)):
            load_data(transformed)

    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)[
        "name"
    ].tolist()
    conn.close()

    expected = ["fato_imoveis", "dim_bairro", "dim_tipo", "normalizacao_stats"]
    for table in expected:
        assert table in tables


# ─────────────────────────────────────────────
# TESTS - PIPELINE COMPLETO
# ─────────────────────────────────────────────
def test_run_pipeline_end_to_end(temp_paths):
    db_path, data_path = temp_paths
    n_samples = 25

    with patch("etl.pipeline.DB_PATH", str(db_path)):
        with patch("etl.pipeline.DATA_PATH", str(data_path)):
            result = run_pipeline(n_samples=n_samples)

    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    df_fato = pd.read_sql("SELECT COUNT(*) as n FROM fato_imoveis", conn)
    conn.close()

    assert df_fato["n"].iloc[0] == n_samples
    assert isinstance(result, dict)


# ─────────────────────────────────────────────
# EXECUÇÃO DIRETA (opcional)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main(["-v", __file__])
