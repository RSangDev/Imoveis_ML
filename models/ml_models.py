"""
Models - Regressão Linear e Clusterização de Bairros
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "imoveis_dw.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data")


def load_from_dw() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM fato_imoveis", conn)
    conn.close()
    return df


# ─────────────────────────────────────────────
# REGRESSÃO LINEAR
# ─────────────────────────────────────────────
def train_regression(df: pd.DataFrame) -> dict:
    """Treina modelo de regressão para prever preço de imóveis."""

    # Encoding de variáveis categóricas
    le_bairro = LabelEncoder()
    le_tipo = LabelEncoder()
    df = df.copy()
    df["bairro_enc"] = le_bairro.fit_transform(df["bairro"])
    df["tipo_enc"] = le_tipo.fit_transform(df["tipo"])

    features = [
        "area_m2",
        "quartos",
        "banheiros",
        "vagas",
        "andar",
        "idade_imovel",
        "condominio",
        "bairro_enc",
        "tipo_enc",
        "renda_media_bairro",
        "idhm_bairro",
    ]

    X = df[features]
    y = df["preco_venda"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Modelo principal: Ridge (regularização L2)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    # Métricas
    metrics = {
        "r2_train": round(r2_score(y_train, y_pred_train), 4),
        "r2_test": round(r2_score(y_test, y_pred_test), 4),
        "rmse_train": round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 0),
        "rmse_test": round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 0),
        "mae_test": round(mean_absolute_error(y_test, y_pred_test), 0),
        "mape_test": round(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100, 2),
    }

    # Coeficientes
    coef_df = pd.DataFrame(
        {
            "feature": features,
            "coeficiente": model.coef_,
            "abs_coef": np.abs(model.coef_),
        }
    ).sort_values("abs_coef", ascending=False)

    # Resultados de predição no teste
    results_df = pd.DataFrame(
        {
            "bairro": df.iloc[y_test.index]["bairro"].values,
            "tipo": df.iloc[y_test.index]["tipo"].values,
            "area_m2": X_test["area_m2"].values,
            "quartos": X_test["quartos"].values,
            "preco_real": y_test.values,
            "preco_previsto": y_pred_test,
            "residuo": y_test.values - y_pred_test,
            "erro_pct": ((y_test.values - y_pred_test) / y_test.values) * 100,
        }
    )

    # Salvar modelo
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler, "le_bairro": le_bairro, "le_tipo": le_tipo},
        os.path.join(MODEL_PATH, "regression_model.pkl"),
    )

    return {
        "model": model,
        "scaler": scaler,
        "le_bairro": le_bairro,
        "le_tipo": le_tipo,
        "metrics": metrics,
        "coef_df": coef_df,
        "results_df": results_df,
        "features": features,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
    }


def predict_price(
    area: float,
    quartos: int,
    banheiros: int,
    vagas: int,
    andar: int,
    idade: int,
    condominio: float,
    bairro: str,
    tipo: str,
    renda_media: float,
    idhm: float,
    model_artifacts: dict,
) -> float:
    """Prediz preço de um imóvel."""
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    le_bairro = model_artifacts["le_bairro"]
    le_tipo = model_artifacts["le_tipo"]

    try:
        bairro_enc = le_bairro.transform([bairro])[0]
    except ValueError:
        bairro_enc = 0
    try:
        tipo_enc = le_tipo.transform([tipo])[0]
    except ValueError:
        tipo_enc = 0

    X = np.array(
        [
            [
                area,
                quartos,
                banheiros,
                vagas,
                andar,
                idade,
                condominio,
                bairro_enc,
                tipo_enc,
                renda_media,
                idhm,
            ]
        ]
    )
    X_s = scaler.transform(X)
    return float(model.predict(X_s)[0])


# ─────────────────────────────────────────────
# CLUSTERIZAÇÃO DE BAIRROS
# ─────────────────────────────────────────────
def cluster_bairros(n_clusters: int = 4) -> dict:
    """Agrupa bairros por perfil socioeconômico."""
    conn = sqlite3.connect(DB_PATH)
    dim_bairro = pd.read_sql("SELECT * FROM dim_bairro", conn)
    conn.close()

    features_cluster = ["preco_m2_medio", "renda_media", "idhm", "preco_medio"]
    X = dim_bairro[features_cluster].fillna(0)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Elbow method data
    inertias = []
    k_range = range(2, min(len(dim_bairro), 8))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_s)
        inertias.append(km.inertia_)

    # Modelo final
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_final.fit(X_s)
    dim_bairro["cluster"] = km_final.labels_

    # Rótulos automáticos por ordem de renda
    cluster_renda = dim_bairro.groupby("cluster")["renda_media"].mean().sort_values()
    label_map = {}
    labels = ["Econômico", "Intermediário", "Alto Padrão", "Luxo"]
    for idx, (cluster_id, _) in enumerate(cluster_renda.items()):
        label_map[cluster_id] = labels[idx] if idx < len(labels) else f"Cluster {idx}"
    dim_bairro["segmento"] = dim_bairro["cluster"].map(label_map)

    # Stats por cluster
    cluster_stats = (
        dim_bairro.groupby("segmento")
        .agg(
            preco_medio=("preco_medio", "mean"),
            renda_media=("renda_media", "mean"),
            idhm_medio=("idhm", "mean"),
            preco_m2_medio=("preco_m2_medio", "mean"),
            qtd_bairros=("bairro", "count"),
        )
        .reset_index()
    )

    return {
        "dim_bairro": dim_bairro,
        "cluster_stats": cluster_stats,
        "inertias": list(inertias),
        "k_range": list(k_range),
        "n_clusters": n_clusters,
    }


if __name__ == "__main__":
    df = load_from_dw()
    result = train_regression(df)
    print("Métricas:", result["metrics"])
    cluster_result = cluster_bairros()
    print(cluster_result["dim_bairro"][["bairro", "segmento"]])
