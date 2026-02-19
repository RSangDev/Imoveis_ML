"""
🏠 Precificação de Imóveis com ML
Aplicação Streamlit - Portfolio Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from etl.pipeline import run_pipeline # noqa
from models.ml_models import (  # noqa
    train_regression,
    predict_price,
    cluster_bairros,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Precificação de Imóveis · ML",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME / CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'); # noqa

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}

[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}

.main-header {
    background: linear-gradient(135deg, #1a2942 0%, #0d1117 50%, #1a1a2e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(
    ellipse at center,
    rgba(88, 166, 255, 0.06) 0%,
    transparent 70%
);

    pointer-events: none;
}

.main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #a5d6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.main-header p {
    color: #8b949e;
    font-size: 1rem;
    margin: 0.5rem 0 0;
}

.kpi-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: all 0.2s ease;
}

.kpi-card:hover {
    border-color: #58a6ff;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(88, 166, 255, 0.12);
}

.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1;
    margin: 0.3rem 0;
}

.kpi-label {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

.kpi-delta {
    font-size: 0.85rem;
    color: #3fb950;
    font-weight: 500;
    margin-top: 0.3rem;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e6edf3;
    border-left: 3px solid #58a6ff;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}

.pipeline-step {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.pipeline-step .step-icon {
    font-size: 1.4rem;
}

.pipeline-step .step-text {
    color: #c9d1d9;
    font-size: 0.88rem;
}

.predict-result {
    background: linear-gradient(135deg, #1a2942, #162032);
    border: 1px solid #58a6ff;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}

.predict-result .price {
    font-size: 2.8rem;
    font-weight: 700;
    color: #58a6ff;
}

.predict-result .price-m2 {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 0.5rem;
}

.segment-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.badge-luxo { 
background: rgba(255, 215, 0, 0.15); 
color: #ffd700; 
border: 1px solid rgba(255,215,0,0.3); }
.badge-alto { 
background: rgba(88, 166, 255, 0.15); 
color: #58a6ff; 
border: 1px solid rgba(88,166,255,0.3); }
.badge-inter { 
background: rgba(63, 185, 80, 0.15); 
color: #3fb950; 
border: 1px solid rgba(63,185,80,0.3); }
.badge-eco { 
background: rgba(139, 148, 158, 0.15); 
color: #8b949e; 
border: 1px solid rgba(139,148,158,0.3); 
}

.metric-good { color: #3fb950; }
.metric-bad { color: #f85149; }

[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
}

.stSelectbox > div, .stSlider { color: #e6edf3; }

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(31, 111, 235, 0.4);
}

div[data-testid="stTabs"] button {
    color: #8b949e;
    font-weight: 500;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff;
    border-bottom-color: #58a6ff;
}

.stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.8)",
    font=dict(family="Inter", color="#c9d1d9", size=12),
    xaxis=dict(
        gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")
    ),
    yaxis=dict(
        gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")
    ),
    margin=dict(t=40, b=40, l=40, r=20),
)

COLOR_PALETTE = ["#58a6ff", "#3fb950", "#ffa657", "#f85149", "#d2a8ff", "#79c0ff"]
SEGMENT_COLORS = {
    "Econômico": "#8b949e",
    "Intermediário": "#3fb950",
    "Alto Padrão": "#58a6ff",
    "Luxo": "#ffd700",
}

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "imoveis_dw.db")


# ─────────────────────────────────────────────
# SESSION STATE / CACHE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    if not os.path.exists(DB_PATH):
        run_pipeline()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM fato_imoveis", conn)
    dim_bairro = pd.read_sql("SELECT * FROM dim_bairro", conn)
    dim_tipo = pd.read_sql("SELECT * FROM dim_tipo", conn)
    conn.close()
    return df, dim_bairro, dim_tipo


@st.cache_data(show_spinner=False)
def get_models(df_hash):
    df, _, _ = get_data()
    reg = train_regression(df)
    clust = cluster_bairros()
    return reg, clust


def fmt_brl(val):
    return f"R$ {val:,.0f}".replace(",", ".")


def fmt_pct(val):
    return f"{val:.1f}%"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size: 2.5rem;'>🏠</div>
        <div style='font-size: 1.1rem; font-weight: 700; color: #e6edf3;
        margin-top: 0.3rem;'>Imóveis ML</div>
        <div style='font-size: 0.75rem; color: #8b949e;'>Portfólio</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("**Navegação**")
    page = st.radio(
        "",
        ["🏠 Dashboard", "🤖 Modelo Preditivo", "🗺️ Segmentação", "⚙️ Pipeline ETL"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("**Configurações**")
    n_samples = st.slider("Nº de registros ETL", 500, 3000, 1500, 250)
    n_clusters = st.slider("Clusters de bairros", 2, 6, 4)

    if st.button("🔄 Reprocessar ETL"):
        st.cache_data.clear()
        with st.spinner("Executando pipeline..."):
            run_pipeline(n_samples=n_samples)
        st.success("ETL concluído!")

    st.markdown("---")
    st.markdown(
        """
    <div style='font-size: 0.75rem; color: #8b949e; line-height: 1.7;'>
    <b style='color:#c9d1d9'>Stack</b><br>
    Python · Pandas · Scikit-learn<br>
    SQLite DW · Plotly · Streamlit<br><br>
    <b style='color:#c9d1d9'>Modelos</b><br>
    Ridge Regression · K-Means<br>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Carregando dados..."):
    if not os.path.exists(DB_PATH):
        run_pipeline(n_samples=n_samples)
    df, dim_bairro, dim_tipo = get_data()

df_hash = len(df)
reg_result, clust_result = get_models(df_hash)

# Adiciona segmento ao df principal
bairro_seg = clust_result["dim_bairro"][["bairro", "segmento"]]
df_enriched = df.merge(bairro_seg, on="bairro", how="left")


# ══════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════
if page == "🏠 Dashboard":

    st.markdown(
        """
    <div class="main-header">
        <h1>🏠 Precificação de Imóveis com ML</h1>
        <p>Pipeline ETL · Regressão Linear
        · Clusterização de Mercado · São Paulo, SP</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # KPIs
    total = len(df)
    preco_med = df["preco_venda"].median()
    preco_m2_med = df["preco_m2"].median()
    r2 = reg_result["metrics"]["r2_test"]
    bairros_n = df["bairro"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        (col1, fmt_brl(preco_med), "Preço Mediano", "Venda"),
        (col2, fmt_brl(preco_m2_med), "Preço/m² Mediano", "São Paulo"),
        (col3, f"{r2:.1%}", "R² do Modelo", "Regressão Ridge"),
        (col4, str(total), "Imóveis", "No dataset"),
        (col5, str(bairros_n), "Bairros", "Mapeados"),
    ]
    for col, val, label, sub in kpis:
        with col:
            st.markdown(
                f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label" style="color:#586069;">{sub}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Distribuição de preços
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(
            '<div class="section-title">Distribuição de Preços por Bairro</div>',
            unsafe_allow_html=True,
        )
        order_bairros = (
            df.groupby("bairro")["preco_venda"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fig = px.box(
            df_enriched,
            x="bairro",
            y="preco_venda",
            color="segmento",
            category_orders={"bairro": order_bairros},
            color_discrete_map=SEGMENT_COLORS,
            labels={"preco_venda": "Preço (R$)", "bairro": "Bairro"},
        )
        fig.update_traces(line_width=1.5)
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", title="Segmento"),
        )
        fig.update_xaxes(tickangle=-40, tickfont_size=10)
        fig.update_yaxes(tickformat=",.0f", tickprefix="R$ ")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown(
            '<div class="section-title">Imóveis por Tipo</div>', unsafe_allow_html=True
        )
        tipo_count = df["tipo"].value_counts().reset_index()
        tipo_count.columns = ["tipo", "count"]
        fig2 = px.pie(
            tipo_count,
            values="count",
            names="tipo",
            color_discrete_sequence=COLOR_PALETTE,
            hole=0.55,
        )
        fig2.update_traces(textinfo="label+percent", textfont_size=11)
        fig2.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
        # margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap correlação
    col_c, col_d = st.columns([2, 3])

    with col_c:
        st.markdown(
            '<div class="section-title">Preço Médio por Segmento</div>',
            unsafe_allow_html=True,
        )
        seg_stats = clust_result["cluster_stats"]
        segs = ["Econômico", "Intermediário", "Alto Padrão", "Luxo"]
        seg_stats_ord = seg_stats[seg_stats["segmento"].isin(segs)].copy()
        seg_stats_ord["ordem"] = seg_stats_ord["segmento"].map(
            {s: i for i, s in enumerate(segs)}
        )
        seg_stats_ord = seg_stats_ord.sort_values("ordem")

        fig3 = go.Figure()
        for _, row in seg_stats_ord.iterrows():
            cor = SEGMENT_COLORS.get(row["segmento"], "#58a6ff")
            fig3.add_trace(
                go.Bar(
                    x=[row["segmento"]],
                    y=[row["preco_medio"]],
                    marker_color=cor,
                    marker_opacity=0.85,
                    name=row["segmento"],
                    showlegend=False,
                    text=[fmt_brl(row["preco_medio"])],
                    textposition="outside",
                    textfont=dict(size=10, color="#c9d1d9"),
                )
            )
        fig3.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title="Preço Médio (R$)")
        fig3.update_yaxes(tickformat=",.0f", tickprefix="R$ ")
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown(
            '<div class="section-title">Área vs. Preço</div>', unsafe_allow_html=True
        )
        sample = df_enriched.sample(min(600, len(df_enriched)), random_state=42)
        fig4 = px.scatter(
            sample,
            x="area_m2",
            y="preco_venda",
            color="segmento",
            size="quartos",
            color_discrete_map=SEGMENT_COLORS,
            opacity=0.65,
            labels={
                "area_m2": "Área (m²)",
                "preco_venda": "Preço (R$)",
                "segmento": "Segmento",
            },
            hover_data=["bairro", "tipo", "quartos"],
        )
        # trendline manual
        m, b = np.polyfit(sample["area_m2"], sample["preco_venda"], 1)
        x_line = np.linspace(sample["area_m2"].min(), sample["area_m2"].max(), 100)
        fig4.add_trace(
            go.Scatter(
                x=x_line,
                y=m * x_line + b,
                mode="lines",
                line=dict(color="#f85149", dash="dash", width=1.5),
                name="Tendência",
                showlegend=True,
            )
        )
        fig4.update_layout(
            **PLOTLY_LAYOUT, height=320, legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        fig4.update_yaxes(tickformat=",.0f", tickprefix="R$ ")
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: MODELO PREDITIVO
# ══════════════════════════════════════════════
elif page == "🤖 Modelo Preditivo":

    st.markdown(
        """
    <div class="main-header">
        <h1>🤖 Modelo de Regressão Linear</h1>
        <p>Ridge Regression para previsão de preço ·
          Análise de resíduos e importância de features</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["📊 Performance", "🔮 Simulador de Preço", "📉 Análise de Resíduos"]
    )

    # ── TAB 1: Performance ──
    with tab1:
        metrics = reg_result["metrics"]

        col1, col2, col3, col4 = st.columns(4)
        kpis_m = [
            (col1, f"{metrics['r2_test']:.4f}", "R² (Teste)", "Var. explicada"),
            (col2, fmt_brl(metrics["rmse_test"]), "RMSE", "Erro médio quadrático"),
            (col3, fmt_brl(metrics["mae_test"]), "MAE", "Erro absoluto médio"),
            (col4, fmt_pct(metrics["mape_test"]), "MAPE", "Erro percentual médio"),
        ]
        for col, val, label, sub in kpis_m:
            with col:
                is_pct = "%" in val
                color = (
                    "#3fb950"
                    if (is_pct and metrics["mape_test"] < 20)
                    or (not is_pct and metrics["r2_test"] > 0.8)
                    else "#58a6ff"
                )
                st.markdown(
                    f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{color};">{val}</div>
                    <div class="kpi-label" style="color:#586069;">{sub}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                '<div class="section-title">Importância das Features'
                ' (|Coeficiente|)</div>',
                unsafe_allow_html=True,
            )
            coef = reg_result["coef_df"]
            fig = px.bar(
                coef,
                x="abs_coef",
                y="feature",
                orientation="h",
                color="abs_coef",
                color_continuous_scale=["#21262d", "#58a6ff"],
                labels={"abs_coef": "|Coeficiente|", "feature": "Feature"},
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=380,
                coloraxis_showscale=False,
                yaxis_categoryorder="total ascending",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown(
                '<div class="section-title">Real vs. Previsto</div>',
                unsafe_allow_html=True,
            )
            res = reg_result["results_df"]
            sample = res.sample(min(400, len(res)), random_state=42)
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=sample["preco_real"],
                    y=sample["preco_previsto"],
                    mode="markers",
                    marker=dict(color="#58a6ff", opacity=0.5, size=5),
                    name="Imóveis",
                )
            )
            lim_max = max(sample["preco_real"].max(), sample["preco_previsto"].max())
            fig2.add_trace(
                go.Scatter(
                    x=[0, lim_max],
                    y=[0, lim_max],
                    mode="lines",
                    line=dict(color="#3fb950", dash="dash", width=2),
                    name="Predição Perfeita",
                )
            )
            fig2.update_layout(
                **PLOTLY_LAYOUT,
                height=380,
                xaxis_title="Preço Real (R$)",
                yaxis_title="Preço Previsto (R$)",
            )
            fig2.update_xaxes(tickformat=",.0f", tickprefix="R$ ")
            fig2.update_yaxes(tickformat=",.0f", tickprefix="R$ ")
            st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 2: Simulador ──
    with tab2:
        st.markdown(
            '<div class="section-title">Configure o Imóvel</div>',
            unsafe_allow_html=True,
        )

        bairros_list = sorted(df["bairro"].unique().tolist())
        tipos_list = sorted(df["tipo"].unique().tolist())

        col1, col2, col3 = st.columns(3)
        with col1:
            sim_bairro = st.selectbox("Bairro", bairros_list, index=0)
            sim_tipo = st.selectbox("Tipo", tipos_list)
            sim_area = st.slider("Área (m²)", 20, 300, 80)
        with col2:
            sim_quartos = st.slider("Quartos", 1, 5, 2)
            sim_banheiros = st.slider("Banheiros", 1, 4, 2)
            sim_vagas = st.slider("Vagas", 0, 4, 1)
        with col3:
            sim_andar = st.slider("Andar", 0, 30, 5)
            sim_idade = st.slider("Idade do imóvel (anos)", 0, 50, 10)
            sim_cond = st.slider("Condomínio (R$)", 0, 5000, 800, 100)

        # Pega renda e idhm do bairro selecionado
        bairro_info = clust_result["dim_bairro"][
            clust_result["dim_bairro"]["bairro"] == sim_bairro
        ]
        renda_bairro = (
            float(bairro_info["renda_media"].values[0]) if len(bairro_info) else 8000
        )
        idhm_bairro = float(bairro_info["idhm"].values[0]) if len(bairro_info) else 0.82
        segmento_bairro = (
            bairro_info["segmento"].values[0] if len(bairro_info) else "Intermediário"
        )

        if st.button("🔮 Prever Preço"):
            preco = predict_price(
                area=sim_area,
                quartos=sim_quartos,
                banheiros=sim_banheiros,
                vagas=sim_vagas,
                andar=sim_andar,
                idade=sim_idade,
                condominio=sim_cond,
                bairro=sim_bairro,
                tipo=sim_tipo,
                renda_media=renda_bairro,
                idhm=idhm_bairro,
                model_artifacts=reg_result,
            )
            preco_m2 = preco / sim_area
            badge_map = {
                "Econômico": "eco",
                "Intermediário": "inter",
                "Alto Padrão": "alto",
                "Luxo": "luxo",
            }
            badge_class = badge_map.get(segmento_bairro, "inter")

            st.markdown(
                f"""
            <div class="predict-result">
                <div style="margin-bottom: 0.5rem;">
                    <span class="segment-badge badge-{badge_class}">{segmento_bairro}
                    </span>
                </div>
                <div class="price">{fmt_brl(preco)}</div>
                <div class="price-m2">≈ {
                    fmt_brl(preco_m2)}/m² · {sim_area}m² · {sim_bairro}</div>
                <div style="margin-top: 1rem; font-size: 0.85rem; color: #8b949e;">
                    Renda média do bairro: {fmt_brl(renda_bairro)}
                    · IDHM: {idhm_bairro:.3f}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Comparação com similares
            similares = df_enriched[
                (df_enriched["bairro"] == sim_bairro)
                & (df_enriched["quartos"] == sim_quartos)
            ]["preco_venda"]
            if len(similares) > 0:
                med_sim = similares.median()
                diff = ((preco - med_sim) / med_sim) * 100
                icon = "📈" if diff > 0 else "📉"
                color = "#ffa657" if diff > 0 else "#3fb950"
                st.markdown(
                    f"""
                <div style='margin-top: 1rem; background: #161b22;
                border: 1px solid #30363d;
                     border-radius: 10px; padding: 1rem; text-align: center;'>
                    <span style='color: #8b949e;'>Mediana de similares no bairro:</span>
                    <strong style='color: #e6edf3;'> {fmt_brl(med_sim)}</strong>
                    <br>
                    <span style='color: {color}; font-size: 1.1rem; font-weight: 600;'>
                        {icon} {diff:+.1f}% em relação à mediana
                    </span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # ── TAB 3: Resíduos ──
    with tab3:
        res = reg_result["results_df"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                '<div class="section-title">Distribuição dos Resíduos</div>',
                unsafe_allow_html=True,
            )
            fig = px.histogram(
                res,
                x="residuo",
                nbins=40,
                color_discrete_sequence=["#58a6ff"],
                opacity=0.8,
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#f85149", line_width=2)
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                xaxis_title="Resíduo (R$)",
                yaxis_title="Frequência",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown(
                '<div class="section-title">Erro % por Bairro</div>',
                unsafe_allow_html=True,
            )
            err_bairro = (
                res.groupby("bairro")["erro_pct"]
                .apply(lambda x: x.abs().mean())
                .sort_values()
            )
            fig2 = px.bar(
                err_bairro, orientation="h", color_discrete_sequence=["#3fb950"]
            )
            fig2.update_layout(
                **PLOTLY_LAYOUT, height=320, xaxis_title="MAPE (%)", yaxis_title=""
            )
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: SEGMENTAÇÃO
# ══════════════════════════════════════════════
elif page == "🗺️ Segmentação":

    st.markdown(
        """
    <div class="main-header">
        <h1>🗺️ Segmentação de Mercado</h1>
        <p>Clusterização K-Means · Perfil socioeconômico dos bairros</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    dim_b = clust_result["dim_bairro"]
    cluster_stats = clust_result["cluster_stats"]

    # Elbow
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown(
            '<div class="section-title">Método do Cotovelo (Elbow)</div>',
            unsafe_allow_html=True,
        )
        fig_elbow = go.Figure()
        fig_elbow.add_trace(
            go.Scatter(
                x=clust_result["k_range"],
                y=clust_result["inertias"],
                mode="lines+markers",
                line=dict(color="#58a6ff", width=2),
                marker=dict(size=8, color="#58a6ff"),
            )
        )
        fig_elbow.add_vline(
            x=n_clusters,
            line_dash="dash",
            line_color="#3fb950",
            annotation_text=f"k={n_clusters}",
            annotation_font_color="#3fb950",
        )
        fig_elbow.update_layout(
            **PLOTLY_LAYOUT,
            height=280,
            xaxis_title="Nº de Clusters (k)",
            yaxis_title="Inércia",
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_b:
        st.markdown(
            '<div class="section-title">Bairros por Segmento</div>',
            unsafe_allow_html=True,
        )
        segs_order = ["Econômico", "Intermediário", "Alto Padrão", "Luxo"]
        dim_b_ord = dim_b[dim_b["segmento"].isin(segs_order)].copy()
        dim_b_ord["ordem"] = dim_b_ord["segmento"].map(
            {s: i for i, s in enumerate(segs_order)}
        )
        dim_b_ord = dim_b_ord.sort_values(["ordem", "renda_media"])

        fig_seg = px.scatter(
            dim_b_ord,
            x="renda_media",
            y="preco_m2_medio",
            color="segmento",
            size="total_imoveis",
            color_discrete_map=SEGMENT_COLORS,
            text="bairro",
            labels={
                "renda_media": "Renda Média (R$)",
                "preco_m2_medio": "Preço/m² Médio (R$)",
            },
        )
        fig_seg.update_traces(textposition="top center", textfont_size=9)
        fig_seg.update_layout(
            **PLOTLY_LAYOUT, height=280, legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        fig_seg.update_xaxes(tickformat=",.0f", tickprefix="R$ ")
        fig_seg.update_yaxes(tickformat=",.0f", tickprefix="R$ ")
        st.plotly_chart(fig_seg, use_container_width=True)

    # Stats por segmento
    st.markdown(
        '<div class="section-title">Perfil dos Segmentos</div>', unsafe_allow_html=True
    )
    cols = st.columns(min(4, len(cluster_stats)))
    for idx, row in cluster_stats[
        cluster_stats["segmento"].isin(segs_order)
    ].iterrows():
        seg = row["segmento"]
        badge_map = {
            "Econômico": "eco",
            "Intermediário": "inter",
            "Alto Padrão": "alto",
            "Luxo": "luxo",
        }
        badge_class = badge_map.get(seg, "inter")
        col = cols[segs_order.index(seg) % len(cols)]
        with col:
            st.markdown(
                f"""
            <div class="kpi-card" style="text-align:left; padding: 1.2rem;">
                <span class="segment-badge
                badge-{badge_class}">{seg}</span>
                <div style="margin-top: 0.8rem; font-size: 0.85rem;
                 color: #8b949e; line-height: 1.9;">
                    <b style="color:#e6edf3;">Preço
                     médio:</b> {fmt_brl(row['preco_medio'])}
                    <br>
                    <b style="color:#e6edf3;">
                    Renda média:</b>
                      {fmt_brl(row['renda_media'])}<br>
                    <b style="color:#e6edf3;">IDHM:</b> {row['idhm_medio']:.3f}<br>
                    <b style="color:#e6edf3;">Preço/m²:
                    </b> {fmt_brl(row['preco_m2_medio'])}<br>
                    <b style="color:#e6edf3;">Bairros:</b> {int(row['qtd_bairros'])}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Tabela de bairros
    st.markdown(
        '<div class="section-title">Tabela de Bairros</div>', unsafe_allow_html=True
    )
    display_cols = [
        "bairro",
        "segmento",
        "preco_medio",
        "preco_m2_medio",
        "renda_media",
        "idhm",
        "total_imoveis",
    ]
    display = dim_b[display_cols].copy()
    display["preco_medio"] = display["preco_medio"].apply(fmt_brl)
    display["preco_m2_medio"] = display["preco_m2_medio"].apply(fmt_brl)
    display["renda_media"] = display["renda_media"].apply(fmt_brl)
    display["idhm"] = display["idhm"].apply(lambda x: f"{x:.3f}")
    display.columns = [
        "Bairro",
        "Segmento",
        "Preço Médio",
        "Preço/m²",
        "Renda Média",
        "IDHM",
        "Imóveis",
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# PAGE: PIPELINE ETL
# ══════════════════════════════════════════════
elif page == "⚙️ Pipeline ETL":

    st.markdown(
        """
    <div class="main-header">
        <h1>⚙️ Pipeline ETL</h1>
        <p>Extração · Transformação · Carga · Data Warehouse SQLite</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Diagrama visual do pipeline
    st.markdown(
        '<div class="section-title">Arquitetura do Pipeline</div>',
        unsafe_allow_html=True,
    )

    steps = [
        (
            "📡",
            "EXTRACT",
            "Simula extração de APIs públicas (Zap Imóveis, Kaggle). "
            "Gera dataset realista com 15 bairros de São Paulo, "
            "variação de preço por região, tipo e características.",
        ),
        (
            "🔧",
            "TRANSFORM",
            "Normalização Z-Score de variáveis numéricas, "
            "criação de features (log_preço, preco_por_quarto, idade_imovel), "
            "encoding de categorias, criação de tabelas dimensionais (Bairro, Tipo).",
        ),
        (
            "💾",
            "LOAD",
            "Carga no Data Warehouse SQLite em schema estrela: "
            "fato_imoveis + dim_bairro + dim_tipo + normalizacao_stats.",
        ),
        (
            "📊",
            "TRANSFORM (ML)",
            "Ridge Regression (L2) sobre features normalizadas. "
            "K-Means para clusterização de bairros por perfil "
            "socioeconômico (renda, IDHM, preço/m²).",
        ),
        (
            "🎯",
            "SERVE",
            "Dashboard Streamlit com KPIs, simulador de preço, "
            "análise de resíduos e mapa de segmentação de mercado.",
        ),
    ]

    for icon, step, desc in steps:
        st.markdown(
            f"""
        <div class="pipeline-step">
            <div class="step-icon">{icon}</div>
            <div>
                <div style="font-size: 0.7rem; font-weight: 700; color: #58a6ff;
                     text-transform: uppercase; letter-spacing: 0.1em;">{step}</div>
                <div class="step-text">{desc}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if icon != "🎯":
            st.markdown(
                "<div style='text-align:center; color:#30363d;" \
                " font-size:1.2rem;'>↓</div>", 
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Stats das tabelas DW
    st.markdown(
        '<div class="section-title">Data Warehouse — Tabelas</div>',
        unsafe_allow_html=True,
    )
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    tables = ["fato_imoveis", "dim_bairro", "dim_tipo", "normalizacao_stats"]
    col_list = st.columns(len(tables))
    for col, table in zip(col_list, tables):
        try:
            count = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn)["n"][0]
            ncols = len(pd.read_sql(f"SELECT * FROM {table} LIMIT 1", conn).columns)
            with col:
                st.markdown(
                    f"""
                <div class="kpi-card">
                    <div class="kpi-label">{table}</div>
                    <div class="kpi-value" style="font-size:1.5rem;">{count}</div>
                    <div class="kpi-label" style="color:#586069;">{ncols} colunas</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        except Exception:
            pass
    conn.close()

    st.markdown("")

    # Schema
    st.markdown(
        '<div class="section-title">Schema — fato_imoveis (amostra)</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # Estatísticas
    st.markdown(
        '<div class="section-title">Estatísticas Descritivas</div>',
        unsafe_allow_html=True,
    )
    num_cols = [
        "area_m2",
        "preco_venda",
        "preco_m2",
        "quartos",
        "banheiros",
        "vagas",
        "idade_imovel",
        "condominio",
    ]
    st.dataframe(
        df[num_cols]
        .describe()
        .round(2)
        .rename(columns={c: c.replace("_", " ") for c in num_cols}),
        use_container_width=True,
    )

    # Tech Stack
    st.markdown(
        '<div class="section-title">Stack Técnico</div>', unsafe_allow_html=True
    )
    stack = {
        "Linguagem": "Python 3.11",
        "ETL": "Pandas · NumPy",
        "DW": "SQLite (schema estrela)",
        "ML": "Scikit-learn (Ridge · K-Means · StandardScaler)",
        "Visualização": "Plotly · Streamlit",
        "Deploy": "Streamlit Community Cloud / Docker",
        "Fonte de Dados": "Simulação de API pública (Kaggle / Zap Imóveis)",
    }
    rows = [[k, v] for k, v in stack.items()]
    st.table(pd.DataFrame(rows, columns=["Componente", "Tecnologia"]))
