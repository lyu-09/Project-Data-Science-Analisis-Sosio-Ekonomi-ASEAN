import os
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

# =====================================================================
# 1. KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Dashboard Pembangunan ASEAN",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 95% !important;
    }

    /* Kartu Metrik */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.82rem !important;
        color: #64748B !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }

    /* Section Header */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        padding-top: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #F1F5F9;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Chart card wrapper */
    .chart-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px 8px 16px;
        margin-bottom: 12px;
    }

    /* Filter bar inside chart */
    .filter-bar {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. KONSTANTA
# =====================================================================
COUNTRY_COLORS = {
    'Indonesia': '#C62828',
    'Malaysia': '#1565C0',
    'Philippines': '#EF6C00',
    'Singapore': '#00796B',
    'Thailand': '#6A1B9A'
}

COUNTRY_FLAGS = {
    'Indonesia': '🇮🇩 Indonesia',
    'Malaysia': '🇲🇾 Malaysia',
    'Philippines': '🇵🇭 Filipina',
    'Singapore': '🇸🇬 Singapura',
    'Thailand': '🇹🇭 Thailand'
}

COUNTRY_CODE_MAP = {
    'IDN': 'Indonesia',
    'MYS': 'Malaysia',
    'PHL': 'Philippines',
    'SGP': 'Singapore',
    'THA': 'Thailand'
}

ALL_COUNTRIES = list(COUNTRY_CODE_MAP.values())

def clean_plot(fig, title="", height=360, showlegend=True):
    """Layout Plotly seragam, bersih, dan nyaman dibaca."""
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>" if title else "",
            'y': 0.96, 'x': 0.01,
            'xanchor': 'left', 'yanchor': 'top',
            'font': {'size': 13, 'color': '#1E293B'}
        },
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=46, b=36),
        height=height,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            title_text="",
            font=dict(size=11, color="#475569")
        ) if showlegend else dict(visible=False),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", tickfont=dict(size=11, color="#64748B")),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", tickfont=dict(size=11, color="#64748B"))
    )
    return fig

def chart_filter(key_prefix, default_countries=None, show_year=True, show_sgp_toggle=False):
    """
    Helper: tampilkan filter negara + tahun di dalam expander kecil.
    Mengembalikan (selected_countries, year_range, hide_sgp).
    """
    if default_countries is None:
        default_countries = ALL_COUNTRIES

    with st.expander("⚙️ Filter Grafik Ini", expanded=False):
        fc1, fc2 = st.columns([1.4, 1] if show_year else [1, 0.01])
        with fc1:
            sel = st.multiselect(
                "Negara:",
                options=ALL_COUNTRIES,
                default=default_countries,
                format_func=lambda x: COUNTRY_FLAGS.get(x, x),
                key=f"{key_prefix}_countries"
            )
        if show_year:
            with fc2:
                yr = st.slider(
                    "Tahun:",
                    min_value=2015, max_value=2024,
                    value=(2015, 2024), step=1,
                    key=f"{key_prefix}_year"
                )
        else:
            yr = (2015, 2024)

        hide_sgp = False
        if show_sgp_toggle:
            hide_sgp = st.checkbox(
                "Pisahkan Singapura (skala proporsional)",
                value=False,
                key=f"{key_prefix}_hidesgp",
                help="Menyembunyikan Singapura agar tren negara berkembang tampak lebih jelas"
            )

    if not sel:
        sel = default_countries
    return sel, yr, hide_sgp

# =====================================================================
# 3. KONEKSI & LOAD DATA DARI DATABASE
# =====================================================================
@st.cache_data(ttl=3600)
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    sqlite_db_path = os.path.join(current_dir, "etl_worldbank.db")

    def init_sqlite_if_needed(db_file):
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dim_country'")
        exists = c.fetchone()
        if not exists:
            sql_init_path = os.path.join(current_dir, "init_sqlite_data.sql")
            if os.path.exists(sql_init_path):
                with open(sql_init_path, "r", encoding="utf-8") as f:
                    c.executescript(f.read())
            conn.commit()
        conn.close()

    engine = None
    db_source = "SQLite Lokal (etl_worldbank.db)"

    try:
        my_engine = create_engine("mysql+mysqlconnector://root:@localhost:3306/etl_worldbank", connect_args={'connect_timeout': 1})
        with my_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine = my_engine
        db_source = "MySQL Server (etl_worldbank)"
    except Exception:
        init_sqlite_if_needed(sqlite_db_path)
        engine = create_engine(f"sqlite:///{sqlite_db_path}")

    with engine.connect() as conn:
        df_c   = pd.read_sql("SELECT * FROM dim_country", conn)
        df_eco = pd.read_sql("SELECT * FROM fact_economic_data", conn)
        df_edu = pd.read_sql("SELECT * FROM fact_education_data", conn)
        df_hlt = pd.read_sql("SELECT * FROM fact_health_data", conn)

    df_edu_c = df_edu.copy()
    for col in ['school_enrollment_primary', 'school_enrollment_secondary', 'govt_expenditure_education', 'literacy_rate_adult']:
        df_edu_c[col] = df_edu_c.groupby('country_code')[col].transform(lambda x: x.bfill().ffill())

    df_hlt_c = df_hlt.copy()
    for col in ['health_expenditure_per_capita', 'infant_mortality_rate']:
        df_hlt_c[col] = df_hlt_c.groupby('country_code')[col].transform(lambda x: x.bfill().ffill())

    df_eco_c = df_eco.copy()

    df_master = (
        df_eco_c
        .merge(df_edu_c.drop(columns=['id', 'created_at'], errors='ignore'), on=['country_code', 'year'], how='inner')
        .merge(df_hlt_c.drop(columns=['id', 'created_at'], errors='ignore'), on=['country_code', 'year'], how='inner')
        .merge(df_c[['country_code', 'country', 'region', 'income_group']], on='country_code', how='inner')
    )

    # HDI Proxy
    inc_min, inc_max = np.log(1000), np.log(100000)
    idx_inc = (np.log(df_master['gdp_per_capita']) - inc_min) / (inc_max - inc_min)
    idx_hlt = (df_master['life_expectancy'] - 60) / (85 - 60)
    edu_comb = (df_master['school_enrollment_secondary'] + df_master['literacy_rate_adult']) / 2
    idx_edu = (edu_comb - 50) / (100 - 50)

    df_master['index_income']    = idx_inc.clip(0, 1)
    df_master['index_health']    = idx_hlt.clip(0, 1)
    df_master['index_education'] = idx_edu.clip(0, 1)
    df_master['hdi_proxy']       = (df_master['index_income'] * df_master['index_health'] * df_master['index_education']) ** (1/3)
    df_master['country_name']    = df_master['country_code'].map(COUNTRY_CODE_MAP)

    return df_c, df_master, db_source

df_country, df_master, db_source = load_data()

# =====================================================================
# 4. HEADER
# =====================================================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("📊 Dashboard Pembangunan ASEAN (2015–2024)")
    st.caption("Visualisasi Terpadu: Ekonomi, Pendidikan, Kesehatan, & Prediksi Machine Learning — Filter tersedia di setiap grafik")
with col_h2:
    st.write("")
    st.caption(f"🔌 **Sumber Data:** {db_source}")

# KPI Ringkas (menggunakan seluruh data master)
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Negara", f"{df_master['country_name'].nunique()} Negara")
kpi2.metric("Rentang Data", "2015 – 2024")
kpi3.metric("Rata-rata GDP/Kapita", f"${df_master['gdp_per_capita'].mean():,.0f}")
kpi4.metric("Rata-rata Harapan Hidup", f"{df_master['life_expectancy'].mean():.1f} Thn")
kpi5.metric("Rata-rata Pengangguran", f"{df_master['unemployment_rate'].mean():.2f}%")

st.write("")

# =====================================================================
# 5. SEKSI 1 — INDIKATOR EKONOMI
# =====================================================================
st.markdown('<div class="section-header">📈 Indikator Ekonomi & Ketenagakerjaan</div>', unsafe_allow_html=True)

col_e1, col_e2 = st.columns([1.5, 1])

# --- Grafik: GDP per Kapita ---
with col_e1:
    with st.container(border=True):
        sel_gdp, yr_gdp, hide_sgp = chart_filter("gdp", show_sgp_toggle=True)
        df_gdp_plot = df_master[
            (df_master['country_name'].isin(sel_gdp)) &
            (df_master['year'] >= yr_gdp[0]) &
            (df_master['year'] <= yr_gdp[1])
        ].copy()
        if hide_sgp:
            df_gdp_plot = df_gdp_plot[df_gdp_plot['country_code'] != 'SGP']

        fig_gdp = px.line(
            df_gdp_plot, x='year', y='gdp_per_capita', color='country_name', markers=True,
            color_discrete_map=COUNTRY_COLORS,
            labels={'gdp_per_capita': 'GDP per Kapita (USD)', 'year': 'Tahun', 'country_name': 'Negara'}
        )
        clean_plot(fig_gdp, "Pertumbuhan GDP Per Kapita (USD)", height=350)
        st.plotly_chart(fig_gdp, use_container_width=True)

# --- Grafik: Pengangguran ---
with col_e2:
    with st.container(border=True):
        sel_unemp, yr_unemp, _ = chart_filter("unemp")
        df_unemp_plot = df_master[
            (df_master['country_name'].isin(sel_unemp)) &
            (df_master['year'] >= yr_unemp[0]) &
            (df_master['year'] <= yr_unemp[1])
        ]
        fig_unemp = px.box(
            df_unemp_plot, x='country_name', y='unemployment_rate',
            color='country_name', points="all",
            color_discrete_map=COUNTRY_COLORS,
            labels={'unemployment_rate': 'Pengangguran (%)', 'country_name': 'Negara'}
        )
        clean_plot(fig_unemp, "Sebaran Tingkat Pengangguran (%)", height=350, showlegend=False)
        st.plotly_chart(fig_unemp, use_container_width=True)

col_e3, col_e4 = st.columns(2)

# --- Grafik: Harapan Hidup ---
with col_e3:
    with st.container(border=True):
        sel_life, yr_life, _ = chart_filter("life")
        df_life_plot = df_master[
            (df_master['country_name'].isin(sel_life)) &
            (df_master['year'] >= yr_life[0]) &
            (df_master['year'] <= yr_life[1])
        ]
        fig_life = px.line(
            df_life_plot, x='year', y='life_expectancy', color='country_name', markers=True,
            color_discrete_map=COUNTRY_COLORS,
            labels={'life_expectancy': 'Harapan Hidup (Tahun)', 'year': 'Tahun', 'country_name': 'Negara'}
        )
        clean_plot(fig_life, "Tren Angka Harapan Hidup (Tahun)", height=330)
        st.plotly_chart(fig_life, use_container_width=True)

# --- Grafik: Total GDP (Bar) ---
with col_e4:
    with st.container(border=True):
        sel_gdpbar, yr_gdpbar, _ = chart_filter("gdpbar")
        df_gdpbar_plot = df_master[
            (df_master['country_name'].isin(sel_gdpbar)) &
            (df_master['year'] >= yr_gdpbar[0]) &
            (df_master['year'] <= yr_gdpbar[1])
        ]
        latest_yr = df_gdpbar_plot['year'].max() if not df_gdpbar_plot.empty else 2024
        df_latest_gdp = df_gdpbar_plot[df_gdpbar_plot['year'] == latest_yr]
        fig_pop_gdp = px.bar(
            df_latest_gdp, x='country_name', y='gdp_billion',
            color='country_name', color_discrete_map=COUNTRY_COLORS,
            text_auto='.1f',
            labels={'gdp_billion': 'Total GDP (Miliar USD)', 'country_name': 'Negara'}
        )
        clean_plot(fig_pop_gdp, f"Total Ukuran Ekonomi GDP (Miliar USD) — Tahun {latest_yr}", height=330, showlegend=False)
        st.plotly_chart(fig_pop_gdp, use_container_width=True)

# =====================================================================
# 6. SEKSI 2 — PENDIDIKAN & KESEHATAN
# =====================================================================
st.markdown('<div class="section-header">🎓 Investasi Pendidikan & Kesehatan</div>', unsafe_allow_html=True)

col_k1, col_k2 = st.columns(2)

# --- Grafik: Belanja Pendidikan ---
with col_k1:
    with st.container(border=True):
        sel_edu, yr_edu, _ = chart_filter("edu")
        df_edu_plot = df_master[
            (df_master['country_name'].isin(sel_edu)) &
            (df_master['year'] >= yr_edu[0]) &
            (df_master['year'] <= yr_edu[1])
        ]
        avg_edu = df_edu_plot.groupby('country_name')['govt_expenditure_education'].mean().reset_index()
        fig_edu = px.bar(
            avg_edu, x='country_name', y='govt_expenditure_education',
            color='country_name', color_discrete_map=COUNTRY_COLORS,
            text_auto='.2f',
            labels={'govt_expenditure_education': 'Belanja (% GDP)', 'country_name': 'Negara'}
        )
        fig_edu.add_hline(y=4.0, line_dash="dash", line_color="#C62828",
                          annotation_text="Target UNESCO (4%)", annotation_position="top right")
        clean_plot(fig_edu, "Rata-rata Alokasi Belanja Pendidikan (% terhadap GDP)", height=340, showlegend=False)
        fig_edu.update_layout(yaxis_range=[0, 6])
        st.plotly_chart(fig_edu, use_container_width=True)

# --- Grafik: Partisipasi Sekolah ---
with col_k2:
    with st.container(border=True):
        sel_enroll, yr_enroll, _ = chart_filter("enroll")
        df_enroll_plot = df_master[
            (df_master['country_name'].isin(sel_enroll)) &
            (df_master['year'] >= yr_enroll[0]) &
            (df_master['year'] <= yr_enroll[1])
        ]
        edu_melt = df_enroll_plot.melt(
            id_vars=['country_name'],
            value_vars=['school_enrollment_primary', 'school_enrollment_secondary'],
            var_name='Jenjang', value_name='Partisipasi (%)'
        )
        edu_melt['Jenjang'] = edu_melt['Jenjang'].map({
            'school_enrollment_primary': 'SD (Primary)',
            'school_enrollment_secondary': 'SMP/SMA (Secondary)'
        })
        avg_enroll = edu_melt.groupby(['country_name', 'Jenjang'])['Partisipasi (%)'].mean().reset_index()
        fig_enroll = px.bar(
            avg_enroll, x='country_name', y='Partisipasi (%)',
            color='Jenjang', barmode='group',
            color_discrete_map={'SD (Primary)': '#1E40AF', 'SMP/SMA (Secondary)': '#93C5FD'},
            labels={'country_name': 'Negara'}
        )
        clean_plot(fig_enroll, "Partisipasi Kasar: SD vs SMP/SMA (Evaluasi Drop-off)", height=340)
        fig_enroll.update_layout(yaxis_range=[0, 120])
        st.plotly_chart(fig_enroll, use_container_width=True)

col_k3, col_k4 = st.columns(2)

# --- Grafik: Belanja Kesehatan ---
with col_k3:
    with st.container(border=True):
        sel_hlt, yr_hlt, _ = chart_filter("hlt")
        df_hlt_plot = df_master[
            (df_master['country_name'].isin(sel_hlt)) &
            (df_master['year'] >= yr_hlt[0]) &
            (df_master['year'] <= yr_hlt[1])
        ]
        avg_hlt = df_hlt_plot.groupby('country_name')['health_expenditure_per_capita'].mean().reset_index()
        fig_hlt = px.bar(
            avg_hlt, x='country_name', y='health_expenditure_per_capita',
            color='country_name', log_y=True, color_discrete_map=COUNTRY_COLORS,
            text_auto='.0f',
            labels={'health_expenditure_per_capita': 'USD / Jiwa (Skala Log)', 'country_name': 'Negara'}
        )
        clean_plot(fig_hlt, "Rata-rata Belanja Kesehatan Per Jiwa (USD - Skala Log)", height=330, showlegend=False)
        st.plotly_chart(fig_hlt, use_container_width=True)

# --- Grafik: Angka Kematian Bayi ---
with col_k4:
    with st.container(border=True):
        sel_inf, yr_inf, _ = chart_filter("inf")
        df_inf_plot = df_master[
            (df_master['country_name'].isin(sel_inf)) &
            (df_master['year'] >= yr_inf[0]) &
            (df_master['year'] <= yr_inf[1])
        ]
        fig_inf = px.line(
            df_inf_plot, x='year', y='infant_mortality_rate',
            color='country_name', markers=True,
            color_discrete_map=COUNTRY_COLORS,
            labels={'infant_mortality_rate': 'Kematian per 1.000', 'year': 'Tahun', 'country_name': 'Negara'}
        )
        clean_plot(fig_inf, "Tren Penurunan Angka Kematian Bayi (per 1.000 Kelahiran)", height=330)
        st.plotly_chart(fig_inf, use_container_width=True)

# =====================================================================
# 7. SEKSI 3 — ANALISIS ANTAR SEKTOR
# =====================================================================
st.markdown('<div class="section-header">🔬 Analisis Antar Sektor & Kualitas Manusia</div>', unsafe_allow_html=True)

col_a1, col_a2 = st.columns(2)

# --- Grafik: Pendidikan vs Pengangguran (Q1) ---
with col_a1:
    with st.container(border=True):
        sel_q1, yr_q1, _ = chart_filter("q1")
        df_q1 = df_master[
            (df_master['country_name'].isin(sel_q1)) &
            (df_master['year'] >= yr_q1[0]) &
            (df_master['year'] <= yr_q1[1])
        ]
        fig_q1 = px.scatter(
            df_q1, x='govt_expenditure_education', y='unemployment_rate',
            color='country_name', color_discrete_map=COUNTRY_COLORS,
            hover_data=['year', 'gdp_per_capita'],
            labels={
                'govt_expenditure_education': 'Belanja Pendidikan (% GDP)',
                'unemployment_rate': 'Pengangguran (%)',
                'country_name': 'Negara'
            }
        )
        fig_q1.update_traces(marker=dict(size=9, opacity=0.85))
        clean_plot(fig_q1, "Belanja Pendidikan vs Pengangguran (Korelasi & Linkage)", height=350)
        st.plotly_chart(fig_q1, use_container_width=True)

# --- Grafik: Kesehatan vs Kematian Bayi (Q2) ---
with col_a2:
    with st.container(border=True):
        sel_q2, yr_q2, _ = chart_filter("q2")
        df_q2 = df_master[
            (df_master['country_name'].isin(sel_q2)) &
            (df_master['year'] >= yr_q2[0]) &
            (df_master['year'] <= yr_q2[1])
        ]
        fig_q2 = px.scatter(
            df_q2, x='health_expenditure_per_capita', y='infant_mortality_rate',
            color='country_name', log_x=True,
            color_discrete_map=COUNTRY_COLORS,
            hover_data=['year', 'life_expectancy'],
            labels={
                'health_expenditure_per_capita': 'Belanja Kesehatan/Jiwa (USD - Log)',
                'infant_mortality_rate': 'Kematian Bayi per 1.000',
                'country_name': 'Negara'
            }
        )
        fig_q2.update_traces(marker=dict(size=9, opacity=0.85))
        clean_plot(fig_q2, "Belanja Kesehatan vs Kematian Bayi (Kurva Efisiensi)", height=350)
        st.plotly_chart(fig_q2, use_container_width=True)

col_a3, col_a4 = st.columns(2)

# --- Grafik: HDI Proxy ---
with col_a3:
    with st.container(border=True):
        sel_hdi, yr_hdi, _ = chart_filter("hdi")
        df_hdi = df_master[
            (df_master['country_name'].isin(sel_hdi)) &
            (df_master['year'] >= yr_hdi[0]) &
            (df_master['year'] <= yr_hdi[1])
        ]
        fig_hdi = px.line(
            df_hdi, x='year', y='hdi_proxy', color='country_name', markers=True,
            color_discrete_map=COUNTRY_COLORS,
            labels={'hdi_proxy': 'Skor HDI Proxy (0-1)', 'year': 'Tahun', 'country_name': 'Negara'}
        )
        fig_hdi.update_layout(yaxis_range=[0.60, 1.0])
        clean_plot(fig_hdi, "Indeks Kualitas Manusia (HDI Proxy 2015–2024)", height=340)
        st.plotly_chart(fig_hdi, use_container_width=True)

# --- Grafik: Time-Lag Pendidikan → GDP ---
with col_a4:
    with st.container(border=True):
        sel_lag, yr_lag, _ = chart_filter("lag")
        lag_rows = []
        for c in df_master['country_name'].unique():
            if c not in sel_lag:
                continue
            c_df = df_master[
                (df_master['country_name'] == c) &
                (df_master['year'] >= yr_lag[0]) &
                (df_master['year'] <= yr_lag[1])
            ].sort_values('year').copy()
            c_df['gdp_lag0'] = c_df['gdp_per_capita']
            c_df['gdp_lag1'] = c_df['gdp_per_capita'].shift(-1)
            c_df['gdp_lag2'] = c_df['gdp_per_capita'].shift(-2)
            c_df['gdp_lag3'] = c_df['gdp_per_capita'].shift(-3)
            for lag in [0, 1, 2, 3]:
                valid = c_df.dropna(subset=['school_enrollment_secondary', f'gdp_lag{lag}'])
                if len(valid) >= 4:
                    r, _ = stats.pearsonr(valid['school_enrollment_secondary'], valid[f'gdp_lag{lag}'])
                    lag_rows.append({'Negara': c, 'Jeda Waktu': f"Jeda {lag} Thn (t+{lag})", 'Korelasi': r})

        df_lag = pd.DataFrame(lag_rows)
        if not df_lag.empty:
            fig_lag = px.bar(
                df_lag, x='Jeda Waktu', y='Korelasi', color='Negara',
                barmode='group', color_discrete_map=COUNTRY_COLORS
            )
            fig_lag.add_hline(y=0, line_color="#94A3B8", line_width=1)
            clean_plot(fig_lag, "Dampak Jeda Waktu Partisipasi SMA terhadap GDP (Time-Lag)", height=340)
            st.plotly_chart(fig_lag, use_container_width=True)
        else:
            st.info("Data tidak cukup untuk menghitung time-lag dengan filter ini.")

# =====================================================================
# 8. SEKSI 4 — PREDIKSI GDP 2025–2027 (ML)
# =====================================================================
st.markdown('<div class="section-header">🤖 Proyeksi Pertumbuhan GDP Per Kapita (2025–2027)</div>', unsafe_allow_html=True)

with st.container(border=True):
    sel_ml, yr_ml, _ = chart_filter("ml", show_year=False)
    st.caption("*Catatan: Filter tahun tidak berlaku di seksi ini — model ML selalu dilatih dari seluruh data historis (2015–2024) untuk akurasi terbaik.*")

    future_years = [2025, 2026, 2027]
    proj_rows = []
    fig_proj = go.Figure()

    for c_name in df_master['country_name'].unique():
        if c_name not in sel_ml:
            continue
        c_sub = df_master[df_master['country_name'] == c_name].sort_values('year')
        X_c = c_sub[['year']].values
        y_c = c_sub['gdp_per_capita'].values

        poly_c = PolynomialFeatures(degree=2)
        X_poly_c = poly_c.fit_transform(X_c)
        m_c = LinearRegression().fit(X_poly_c, y_c)

        X_fut = np.array(future_years).reshape(-1, 1)
        y_fut = m_c.predict(poly_c.transform(X_fut))

        fig_proj.add_trace(go.Scatter(
            x=c_sub['year'], y=y_c, mode='lines+markers',
            name=f"{c_name}",
            line=dict(color=COUNTRY_COLORS.get(c_name, '#333333'), width=2)
        ))
        fig_proj.add_trace(go.Scatter(
            x=[c_sub['year'].iloc[-1]] + future_years,
            y=[y_c[-1]] + list(y_fut),
            mode='lines+markers', name=f"{c_name} (Prediksi)",
            line=dict(color=COUNTRY_COLORS.get(c_name, '#333333'), width=2, dash='dash')
        ))

        for yr_val, val in zip(future_years, y_fut):
            proj_rows.append({'Negara': c_name, 'Tahun': yr_val, 'Prediksi GDP/Kapita (USD)': round(val, 2)})

    fig_proj.add_vline(x=2024.5, line_width=1, line_dash="dot", line_color="#94A3B8", annotation_text="Batas Prediksi")
    clean_plot(fig_proj, "Kurva Proyeksi Historis & Prediksi 2025–2027", height=380)

    col_p1, col_p2 = st.columns([1.6, 1])
    with col_p1:
        st.plotly_chart(fig_proj, use_container_width=True)
    with col_p2:
        df_proj = pd.DataFrame(proj_rows)
        if not df_proj.empty:
            st.write("**Tabel Angka Proyeksi:**")
            pivot_proj = df_proj.pivot(index='Negara', columns='Tahun', values='Prediksi GDP/Kapita (USD)')
            st.dataframe(pivot_proj.applymap(lambda x: f"${x:,.2f}"), use_container_width=True)
            csv_data = df_proj.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil Proyeksi (CSV)",
                data=csv_data,
                file_name='proyeksi_gdp_asean_2025_2027.csv',
                mime='text/csv'
            )

# =====================================================================
# 9. FOOTER
# =====================================================================
st.write("")
st.caption("Dashboard Sains Data Pembangunan ASEAN | Sumber: Bank Dunia (World Bank Open Data) | Terhubung Database Relasional")
