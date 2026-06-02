import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .stApp {
        background: #0a0e1a;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background: #0d1120 !important;
        border-right: 1px solid #1e2a45;
    }
    section[data-testid="stSidebar"] * {
        color: #a0aec0 !important;
    }
    .stSelectbox label, .stSlider label, .stMultiSelect label {
        color: #a0aec0 !important;
        font-size: 12px !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }
    .stSelectbox > div > div {
        background: #111827 !important;
        border: 1px solid #1e2a45 !important;
        color: #e2e8f0 !important;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #1e2a45;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-label {
        font-size: 11px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-delta {
        font-size: 12px;
        margin-top: 4px;
    }
    .delta-up { color: #10b981; }
    .delta-down { color: #ef4444; }
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #64748b;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2a45;
    }
    .insight-card {
        background: #111827;
        border-left: 3px solid #8b5cf6;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .insight-card.teal { border-left-color: #14b8a6; }
    .insight-card.red { border-left-color: #ef4444; }
    .insight-card.amber { border-left-color: #f59e0b; }
    .insight-title {
        font-size: 14px;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 6px;
    }
    .insight-body {
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.5;
    }
    .boot-box {
        background: #0d1120;
        border: 1px solid #1e2a45;
        border-radius: 10px;
        padding: 20px 24px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #10b981;
        line-height: 2;
    }
    .hero-title {
        font-size: 52px;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .hero-purple { color: #8b5cf6; }
    .hero-white { color: #e2e8f0; }
    .hero-teal { color: #14b8a6; }
    .badge {
        display: inline-block;
        background: #1e2a45;
        border: 1px solid #2d3f5e;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 11px;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin: 4px;
    }
    .page-title {
        font-size: 32px;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .page-sub {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 24px;
    }
    div[data-testid="stHorizontalBlock"] { gap: 12px; }
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }
    footer { display: none; }
    #MainMenu { display: none; }
    header { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://owid-public.owid.io/data/energy/owid-energy-data.csv"
    df = pd.read_csv(url)
    df = df[~df['country'].isin([
        'World', 'Asia', 'Europe', 'Africa', 'North America',
        'South America', 'Oceania', 'High-income countries',
        'Low-income countries', 'Upper-middle-income countries',
        'Lower-middle-income countries', 'European Union (27)'
    ])]
    return df

PLOTLY_THEME = dict(
    plot_bgcolor='#111827',
    paper_bgcolor='#111827',
    font_color='#94a3b8',
    font_size=12,
    title_font_color='#e2e8f0',
    title_font_size=15,
    legend_bgcolor='#111827',
    legend_bordercolor='#1e2a45',
    gridcolor='#1e2a45',
    colorway=['#8b5cf6','#14b8a6','#f59e0b','#ef4444','#3b82f6','#10b981','#f97316']
)

def apply_theme(fig):
    fig.update_layout(
        plot_bgcolor=PLOTLY_THEME['plot_bgcolor'],
        paper_bgcolor=PLOTLY_THEME['paper_bgcolor'],
        font=dict(color=PLOTLY_THEME['font_color'], size=PLOTLY_THEME['font_size']),
        title_font=dict(color=PLOTLY_THEME['title_font_color'], size=PLOTLY_THEME['title_font_size']),
        legend=dict(bgcolor=PLOTLY_THEME['legend_bgcolor'], bordercolor=PLOTLY_THEME['legend_bordercolor'], borderwidth=1),
        colorway=PLOTLY_THEME['colorway']
    )
    fig.update_xaxes(gridcolor=PLOTLY_THEME['gridcolor'], zerolinecolor=PLOTLY_THEME['gridcolor'])
    fig.update_yaxes(gridcolor=PLOTLY_THEME['gridcolor'], zerolinecolor=PLOTLY_THEME['gridcolor'])
    return fig

# ── Sidebar Navigation ────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:22px; font-weight:700; color:#8b5cf6; letter-spacing:0.05em;'>NEXUS</div>
        <div style='font-size:10px; letter-spacing:0.2em; color:#475569; margin-top:2px;'>ENERGY COMMAND CENTER</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border-top:1px solid #1e2a45; margin:10px 0 16px 0;'></div>", unsafe_allow_html=True)

    page = st.selectbox("", [
        "🏠  Mission Control",
        "📊  Executive Overview",
        "⚡  Energy Consumption",
        "🌱  Renewables Intelligence",
        "🔥  Fossil Fuel Analytics",
        "💨  CO₂ & Emissions",
        "🔮  Predictive Analytics",
        "🌍  Country Deep Dive",
        "📡  Real-Time Monitor",
    ], label_visibility="collapsed")

    st.markdown("<div style='border-top:1px solid #1e2a45; margin:16px 0 12px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px; letter-spacing:0.15em; color:#334155; text-transform:uppercase; padding: 0 4px;'>Navigation Matrix</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px; color:#475569; padding: 8px 4px; line-height:2;'>
    🏠 Mission Control<br>
    📊 Executive Overview<br>
    ⚡ Energy Consumption<br>
    🌱 Renewables Intelligence<br>
    🔥 Fossil Fuel Analytics<br>
    💨 CO₂ & Emissions<br>
    🔮 Predictive Analytics<br>
    🌍 Country Deep Dive<br>
    📡 Real-Time Monitor
    </div>
    """, unsafe_allow_html=True)

# Load data
with st.spinner("Loading OWID Energy Intelligence..."):
    df = load_data()

latest_year = int(df['year'].max())
countries_list = sorted(df['country'].dropna().unique().tolist())

# ══════════════════════════════════════════════════════════
# PAGE 1 — MISSION CONTROL
# ══════════════════════════════════════════════════════════
if "Mission Control" in page:
    st.markdown("""
    <div style='margin-bottom:24px;'>
        <div class='hero-title'>
            <span class='hero-purple'>NEXUS</span><span class='hero-white'> AI</span><br>
            <span class='hero-teal'>Energy</span><span class='hero-white'> Analytics</span>
        </div>
        <div style='font-size:16px; color:#64748b; margin-bottom:16px;'>
            Global Energy Consumption, Renewables & Emissions Intelligence
        </div>
        <span class='badge'>ENERGY INTELLIGENCE</span>
        <span class='badge'>RENEWABLES TRACKING</span>
        <span class='badge'>PREDICTIVE ML</span>
        <span class='badge'>REAL-TIME MONITOR</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='boot-box'>
        nexus:// boot sequence complete<br>
        nexus:// energy telemetry linked — {latest_year} data loaded<br>
        nexus:// renewables intelligence graph online<br>
        nexus:// predictive models available in analytics bay<br>
        nexus:// {len(countries_list)} countries indexed and ready
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>System Intelligence</div>", unsafe_allow_html=True)

    latest = df[df['year'] == latest_year]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = latest['primary_energy_consumption'].sum()
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Energy (TWh)</div><div class='metric-value'>{val/1000:,.0f}K</div></div>", unsafe_allow_html=True)
    with c2:
        val = latest['renewables_share_energy'].mean()
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Renewables %</div><div class='metric-value'>{val:.1f}%</div></div>", unsafe_allow_html=True)
    with c3:
        val = latest['fossil_share_energy'].mean()
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Fossil %</div><div class='metric-value'>{val:.1f}%</div></div>", unsafe_allow_html=True)
    with c4:
        val = latest['co2_per_capita'].mean()
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg CO₂/Capita (t)</div><div class='metric-value'>{val:.1f}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Key Intelligence</div>", unsafe_allow_html=True)

    top_renew = latest.nlargest(1, 'renewables_share_energy')[['country','renewables_share_energy']].values
    top_co2 = latest.nlargest(1, 'co2_per_capita')[['country','co2_per_capita']].values
    top_consumer = latest.nlargest(1, 'primary_energy_consumption')[['country','primary_energy_consumption']].values

    st.markdown(f"""
    <div class='insight-card teal'>
        <div class='insight-title'>Renewables leader detected</div>
        <div class='insight-body'>{top_renew[0][0]} leads with {top_renew[0][1]:.1f}% renewable energy share in {latest_year}.</div>
    </div>
    <div class='insight-card red'>
        <div class='insight-title'>High CO₂ intensity flagged</div>
        <div class='insight-body'>{top_co2[0][0]} records highest CO₂ per capita at {top_co2[0][1]:.1f} tonnes per person.</div>
    </div>
    <div class='insight-card amber'>
        <div class='insight-title'>Largest energy consumer</div>
        <div class='insight-body'>{top_consumer[0][0]} consumes {top_consumer[0][1]:,.0f} TWh — dominating global energy demand.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Analytics Pages</div>", unsafe_allow_html=True)
    pages_info = [
        ("📊", "Executive Overview", "Global KPIs and summary charts"),
        ("⚡", "Energy Consumption", "Trends across countries and time"),
        ("🌱", "Renewables Intelligence", "Solar, wind, hydro growth analysis"),
        ("🔥", "Fossil Fuel Analytics", "Coal, oil, gas consumption tracking"),
        ("💨", "CO₂ & Emissions", "Carbon intensity and emissions data"),
        ("🔮", "Predictive Analytics", "ML-powered forecasting models"),
        ("🌍", "Country Deep Dive", "Single-country full profile"),
        ("📡", "Real-Time Monitor", "Live ranking and comparison"),
    ]
    col_a, col_b = st.columns(2)
    for i, (icon, name, desc) in enumerate(pages_info):
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(f"""
            <div style='background:#111827; border:1px solid #1e2a45; border-radius:10px; padding:14px; margin-bottom:10px;'>
                <div style='font-size:18px; margin-bottom:4px;'>{icon}</div>
                <div style='font-size:13px; font-weight:600; color:#e2e8f0;'>MODULE {i+1:02d} — {name.upper()}</div>
                <div style='font-size:12px; color:#475569; margin-top:2px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════
elif "Executive Overview" in page:
    st.markdown("<div class='page-title'>Executive Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Global energy intelligence summary — all indicators at a glance</div>", unsafe_allow_html=True)

    latest = df[df['year'] == latest_year]
    prev = df[df['year'] == latest_year - 1]

    def delta(col):
        cur = latest[col].mean()
        prv = prev[col].mean()
        return cur, ((cur - prv) / prv * 100) if prv else 0

    metrics = [
        ("Primary Energy (TWh)", 'primary_energy_consumption', True, "K"),
        ("Renewables Share %", 'renewables_share_energy', True, "%"),
        ("Fossil Fuels Share %", 'fossil_share_energy', False, "%"),
        ("CO₂ per Capita (t)", 'co2_per_capita', False, ""),
        ("Nuclear Share %", 'nuclear_share_energy', True, "%"),
        ("Solar Share %", 'solar_share_energy', True, "%"),
        ("Wind Share %", 'wind_share_elec', True, "%"),
        ("Energy per Capita", 'energy_per_capita', True, ""),
    ]

    cols = st.columns(4)
    for i, (label, col, up_good, suffix) in enumerate(metrics):
        if col in df.columns:
            val, d = delta(col)
            if suffix == "K":
                display = f"{val/1000:,.0f}K"
            elif suffix == "%":
                display = f"{val:.1f}%"
            else:
                display = f"{val:,.1f}"
            good = (d > 0 and up_good) or (d < 0 and not up_good)
            arrow = "▲" if d > 0 else "▼"
            color = "#10b981" if good else "#ef4444"
            with cols[i % 4]:
                st.markdown(f"""
                <div class='metric-card' style='margin-bottom:12px;'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value'>{display}</div>
                    <div class='metric-delta' style='color:{color};'>{arrow} {abs(d):.1f}% vs {latest_year-1}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Global Energy Mix Trend</div>", unsafe_allow_html=True)
    world_cols = ['year','coal_share_energy','oil_share_energy','gas_share_energy',
                  'nuclear_share_energy','renewables_share_energy']
    world = df[world_cols].groupby('year').mean().reset_index()
    world = world[world['year'] >= 1990]

    fig = go.Figure()
    colors = ['#ef4444','#f97316','#f59e0b','#8b5cf6','#10b981']
    names = ['Coal','Oil','Gas','Nuclear','Renewables']
    for i, (col, name) in enumerate(zip(['coal_share_energy','oil_share_energy','gas_share_energy','nuclear_share_energy','renewables_share_energy'], names)):
        fig.add_trace(go.Scatter(x=world['year'], y=world[col], name=name,
                                  stackgroup='one', line=dict(color=colors[i]), fillcolor=colors[i]+"40"))
    fig.update_layout(title="Global Energy Mix (1990–present)", height=350,
                      xaxis_title="Year", yaxis_title="Share (%)")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>Top 10 Energy Consumers</div>", unsafe_allow_html=True)
        top10 = df[df['year'] == latest_year].nlargest(10, 'primary_energy_consumption')[['country','primary_energy_consumption']].dropna()
        fig2 = px.bar(top10, x='primary_energy_consumption', y='country', orientation='h',
                      color='primary_energy_consumption', color_continuous_scale=['#1e2a45','#8b5cf6'],
                      labels={'primary_energy_consumption':'TWh','country':''})
        fig2.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Top 10 Renewables Leaders</div>", unsafe_allow_html=True)
        top_r = df[df['year'] == latest_year].nlargest(10, 'renewables_share_energy')[['country','renewables_share_energy']].dropna()
        fig3 = px.bar(top_r, x='renewables_share_energy', y='country', orientation='h',
                      color='renewables_share_energy', color_continuous_scale=['#1e2a45','#14b8a6'],
                      labels={'renewables_share_energy':'%','country':''})
        fig3.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — ENERGY CONSUMPTION
# ══════════════════════════════════════════════════════════
elif "Energy Consumption" in page:
    st.markdown("<div class='page-title'>Energy Consumption</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Primary energy consumption trends across countries and time</div>", unsafe_allow_html=True)

    selected = st.multiselect("Select Countries", countries_list,
                               default=["Pakistan","India","China","United States","Germany"])
    yr = st.slider("Year Range", 1970, latest_year, (2000, latest_year))

    filt = df[df['country'].isin(selected) & df['year'].between(yr[0], yr[1])]

    fig = px.line(filt, x='year', y='primary_energy_consumption', color='country',
                  labels={'primary_energy_consumption':'TWh','year':'Year'},
                  title="Primary Energy Consumption Over Time")
    fig.update_traces(line_width=2)
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(filt, x='year', y='energy_per_capita', color='country', barmode='group',
                      title="Energy per Capita (kWh/person)",
                      labels={'energy_per_capita':'kWh','year':'Year'})
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        latest_filt = filt[filt['year'] == filt['year'].max()][['country','primary_energy_consumption']].dropna()
        fig3 = px.pie(latest_filt, names='country', values='primary_energy_consumption',
                      title=f"Energy Share Among Selected ({latest_year})",
                      hole=0.4)
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Consumption Data Table</div>", unsafe_allow_html=True)
    show_cols = ['country','year','primary_energy_consumption','energy_per_capita','population']
    st.dataframe(filt[show_cols].dropna().sort_values(['country','year'], ascending=[True,False]),
                 use_container_width=True, height=250)

# ══════════════════════════════════════════════════════════
# PAGE 4 — RENEWABLES INTELLIGENCE
# ══════════════════════════════════════════════════════════
elif "Renewables" in page:
    st.markdown("<div class='page-title'>Renewables Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Solar, wind, hydro and total renewable energy growth analysis</div>", unsafe_allow_html=True)

    selected = st.multiselect("Select Countries", countries_list,
                               default=["Pakistan","India","Germany","United States","China"])
    yr = st.slider("Year Range", 1990, latest_year, (2000, latest_year))
    filt = df[df['country'].isin(selected) & df['year'].between(yr[0], yr[1])]

    fig = px.area(filt, x='year', y='renewables_share_energy', color='country',
                  title="Renewables Share of Total Energy (%)",
                  labels={'renewables_share_energy':'%','year':'Year'})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig2 = px.line(filt, x='year', y='solar_share_energy', color='country',
                       title="Solar Share (%)", labels={'solar_share_energy':'%'})
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.line(filt, x='year', y='wind_share_energy', color='country',
                       title="Wind Share (%)", labels={'wind_share_energy':'%'})
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)
    with col3:
        fig4 = px.line(filt, x='year', y='hydro_share_energy', color='country',
                       title="Hydro Share (%)", labels={'hydro_share_energy':'%'})
        apply_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-header'>Renewables Rankings — Latest Year</div>", unsafe_allow_html=True)
    top20 = df[df['year'] == latest_year].nlargest(20, 'renewables_share_energy')[['country','renewables_share_energy','solar_share_energy','wind_share_energy','hydro_share_energy']].dropna()
    fig5 = px.bar(top20, x='country', y=['solar_share_energy','wind_share_energy','hydro_share_energy'],
                  title="Top 20 Countries — Renewables Breakdown",
                  labels={'value':'%','variable':'Source'},
                  color_discrete_map={'solar_share_energy':'#f59e0b','wind_share_energy':'#14b8a6','hydro_share_energy':'#3b82f6'})
    apply_theme(fig5)
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 5 — FOSSIL FUEL ANALYTICS
# ══════════════════════════════════════════════════════════
elif "Fossil" in page:
    st.markdown("<div class='page-title'>Fossil Fuel Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Coal, oil, and gas consumption and production tracking</div>", unsafe_allow_html=True)

    selected = st.multiselect("Select Countries", countries_list,
                               default=["Pakistan","India","China","United States","Russia"])
    yr = st.slider("Year Range", 1970, latest_year, (1990, latest_year))
    filt = df[df['country'].isin(selected) & df['year'].between(yr[0], yr[1])]

    fig = px.line(filt, x='year', y='fossil_share_energy', color='country',
                  title="Fossil Fuel Share of Total Energy (%)",
                  labels={'fossil_share_energy':'%','year':'Year'})
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.line(filt, x='year', y='coal_share_energy', color='country',
                       title="Coal Share (%)", labels={'coal_share_energy':'%'})
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.line(filt, x='year', y='oil_share_energy', color='country',
                       title="Oil Share (%)", labels={'oil_share_energy':'%'})
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Highest Fossil Dependency — Latest Year</div>", unsafe_allow_html=True)
    top_fossil = df[df['year'] == latest_year].nlargest(15, 'fossil_share_energy')[['country','fossil_share_energy','coal_share_energy','oil_share_energy','gas_share_energy']].dropna()
    fig4 = px.bar(top_fossil, x='fossil_share_energy', y='country', orientation='h',
                  color='fossil_share_energy', color_continuous_scale=['#1e2a45','#ef4444'],
                  title="Top 15 Fossil-Dependent Countries",
                  labels={'fossil_share_energy':'%','country':''})
    fig4.update_layout(height=400, coloraxis_showscale=False)
    apply_theme(fig4)
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 6 — CO2 & EMISSIONS
# ══════════════════════════════════════════════════════════
elif "CO₂" in page:
    st.markdown("<div class='page-title'>CO₂ & Emissions</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Carbon intensity, emissions per capita and greenhouse gas tracking</div>", unsafe_allow_html=True)

    selected = st.multiselect("Select Countries", countries_list,
                               default=["Pakistan","India","China","United States","Germany","Saudi Arabia"])
    yr = st.slider("Year Range", 1990, latest_year, (2000, latest_year))
    filt = df[df['country'].isin(selected) & df['year'].between(yr[0], yr[1])]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(filt, x='year', y='co2_per_capita', color='country',
                      title="CO₂ per Capita (tonnes)",
                      labels={'co2_per_capita':'Tonnes','year':'Year'})
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.line(filt, x='year', y='greenhouse_gas_emissions', color='country',
                       title="Greenhouse Gas Emissions (TWh)",
                       labels={'greenhouse_gas_emissions':'TWh','year':'Year'})
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>CO₂ vs Energy Consumption — Bubble Chart</div>", unsafe_allow_html=True)
    bubble = filt[filt['year'] == filt['year'].max()].dropna(subset=['co2_per_capita','energy_per_capita','population'])
    fig3 = px.scatter(bubble, x='energy_per_capita', y='co2_per_capita',
                      size='population', color='country', hover_name='country',
                      title=f"CO₂ per Capita vs Energy per Capita ({latest_year})",
                      labels={'energy_per_capita':'Energy per Capita (kWh)','co2_per_capita':'CO₂ per Capita (t)'})
    apply_theme(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Highest CO₂ Emitters — Latest Year</div>", unsafe_allow_html=True)
    top_co2 = df[df['year'] == latest_year].nlargest(15, 'co2_per_capita')[['country','co2_per_capita']].dropna()
    fig4 = px.bar(top_co2, x='co2_per_capita', y='country', orientation='h',
                  color='co2_per_capita', color_continuous_scale=['#1e2a45','#ef4444'],
                  labels={'co2_per_capita':'Tonnes per Person','country':''})
    fig4.update_layout(height=380, coloraxis_showscale=False)
    apply_theme(fig4)
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 7 — PREDICTIVE ANALYTICS
# ══════════════════════════════════════════════════════════
elif "Predictive" in page:
    st.markdown("<div class='page-title'>Predictive Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>ML-powered energy forecasting models with confidence intervals</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        pred_country = st.selectbox("Country", countries_list, index=countries_list.index("Pakistan") if "Pakistan" in countries_list else 0)
    with col2:
        pred_metric = st.selectbox("Metric to Forecast", [
            ("Primary Energy Consumption", "primary_energy_consumption"),
            ("Renewables Share %", "renewables_share_energy"),
            ("CO₂ per Capita", "co2_per_capita"),
            ("Solar Share %", "solar_share_energy"),
            ("Fossil Fuels Share %", "fossil_share_energy"),
        ], format_func=lambda x: x[0])
    with col3:
        horizon = st.slider("Forecast Horizon (years)", 5, 25, 10)

    metric_col = pred_metric[1]
    metric_name = pred_metric[0]

    cdata = df[(df['country'] == pred_country) & df[metric_col].notna()].sort_values('year')

    if len(cdata) >= 10:
        X = cdata[['year']].values
        y = cdata[metric_col].values

        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('lr', LinearRegression())
        ])
        model.fit(X, y)
        r2 = model.score(X, y)

        future_yrs = np.arange(latest_year + 1, latest_year + horizon + 1).reshape(-1, 1)
        future_preds = model.predict(future_yrs)
        hist_preds = model.predict(X)

        residuals = y - hist_preds
        std = residuals.std()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cdata['year'], y=y, name='Historical',
                                  line=dict(color='#8b5cf6', width=2)))
        fig.add_trace(go.Scatter(x=cdata['year'], y=hist_preds, name='Model Fit',
                                  line=dict(color='#64748b', width=1, dash='dot')))
        fig.add_trace(go.Scatter(
            x=np.concatenate([future_yrs.flatten(), future_yrs.flatten()[::-1]]),
            y=np.concatenate([future_preds + 2*std, (future_preds - 2*std)[::-1]]),
            fill='toself', fillcolor='rgba(20,184,166,0.1)',
            line=dict(color='rgba(0,0,0,0)'), name='95% Confidence'))
        fig.add_trace(go.Scatter(x=future_yrs.flatten(), y=future_preds, name='Forecast',
                                  line=dict(color='#14b8a6', width=2, dash='dash')))
        fig.update_layout(title=f"{metric_name} Forecast — {pred_country} (→{latest_year+horizon})",
                          xaxis_title="Year", yaxis_title=metric_name, height=400)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Model R² Score</div><div class='metric-value' style='font-size:22px;'>{r2:.3f}</div><div class='metric-delta {'delta-up' if r2>0.8 else 'delta-down'}'>{'Good Fit ✅' if r2>0.8 else 'Moderate ⚠️'}</div></div>", unsafe_allow_html=True)
        with c2:
            cur_val = cdata[metric_col].iloc[-1]
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Current ({latest_year})</div><div class='metric-value' style='font-size:22px;'>{cur_val:,.1f}</div></div>", unsafe_allow_html=True)
        with c3:
            mid_val = future_preds[len(future_preds)//2]
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Mid Forecast</div><div class='metric-value' style='font-size:22px;'>{mid_val:,.1f}</div></div>", unsafe_allow_html=True)
        with c4:
            end_val = future_preds[-1]
            st.markdown(f"<div class='metric-card'><div class='metric-label'>End Forecast ({latest_year+horizon})</div><div class='metric-value' style='font-size:22px;'>{end_val:,.1f}</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Forecast Table</div>", unsafe_allow_html=True)
        forecast_df = pd.DataFrame({
            'Year': future_yrs.flatten(),
            'Forecast': future_preds.round(2),
            'Lower Bound': (future_preds - 2*std).round(2),
            'Upper Bound': (future_preds + 2*std).round(2)
        })
        st.dataframe(forecast_df, use_container_width=True, height=250)
    else:
        st.warning(f"Not enough data for {pred_country}. Please select another country.")

# ══════════════════════════════════════════════════════════
# PAGE 8 — COUNTRY DEEP DIVE
# ══════════════════════════════════════════════════════════
elif "Country Deep Dive" in page:
    st.markdown("<div class='page-title'>Country Deep Dive</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Full energy profile for a single country</div>", unsafe_allow_html=True)

    country = st.selectbox("Select Country", countries_list,
                            index=countries_list.index("Pakistan") if "Pakistan" in countries_list else 0)
    cdata = df[df['country'] == country].sort_values('year')
    latest_c = cdata[cdata['year'] == cdata['year'].max()].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Primary Energy (TWh)", 'primary_energy_consumption', "{:,.0f}"),
        ("Renewables Share", 'renewables_share_energy', "{:.1f}%"),
        ("CO₂ per Capita", 'co2_per_capita', "{:.2f}t"),
        ("Fossil Share", 'fossil_share_energy', "{:.1f}%"),
    ]
    for i, (label, col, fmt) in enumerate(kpis):
        val = latest_c.get(col, np.nan)
        display = fmt.format(val) if not pd.isna(val) else "N/A"
        with [c1,c2,c3,c4][i]:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='font-size:24px;'>{display}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Energy Mix Trend</div>", unsafe_allow_html=True)
    fig = px.area(cdata, x='year',
                  y=['coal_share_energy','oil_share_energy','gas_share_energy','nuclear_share_energy','renewables_share_energy'],
                  title=f"{country} — Energy Mix Over Time (%)",
                  labels={'value':'%','year':'Year','variable':'Source'},
                  color_discrete_map={
                      'coal_share_energy':'#ef4444',
                      'oil_share_energy':'#f97316',
                      'gas_share_energy':'#f59e0b',
                      'nuclear_share_energy':'#8b5cf6',
                      'renewables_share_energy':'#10b981'
                  })
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.line(cdata, x='year', y=['solar_share_energy','wind_share_energy','hydro_share_energy'],
                       title="Renewables Breakdown (%)",
                       color_discrete_map={'solar_share_energy':'#f59e0b','wind_share_energy':'#14b8a6','hydro_share_energy':'#3b82f6'})
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.line(cdata, x='year', y='co2_per_capita',
                       title="CO₂ per Capita (tonnes)", labels={'co2_per_capita':'tonnes'})
        fig3.update_traces(line_color='#ef4444')
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 9 — REAL-TIME MONITOR
# ══════════════════════════════════════════════════════════
elif "Real-Time" in page:
    st.markdown("<div class='page-title'>Real-Time Monitor</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Live rankings, comparisons, and global leaderboards</div>", unsafe_allow_html=True)

    metric_choice = st.selectbox("Rank by", [
        ("Primary Energy Consumption (TWh)", "primary_energy_consumption"),
        ("Renewables Share (%)", "renewables_share_energy"),
        ("CO₂ per Capita (tonnes)", "co2_per_capita"),
        ("Solar Share (%)", "solar_share_energy"),
        ("Wind Share (%)", "wind_share_elec"),
        ("Fossil Fuels Share (%)", "fossil_share_energy"),
    ], format_func=lambda x: x[0])

    top_n = st.slider("Top N Countries", 10, 50, 20)
    mcol = metric_choice[1]
    mname = metric_choice[0]

    latest = df[df['year'] == latest_year][['country', mcol]].dropna()
    top = latest.nlargest(top_n, mcol).reset_index(drop=True)
    top.index = top.index + 1

    fig = px.bar(top, x=mcol, y='country', orientation='h',
                 color=mcol, color_continuous_scale=['#1e2a45','#8b5cf6','#14b8a6'],
                 title=f"Top {top_n} Countries — {mname} ({latest_year})",
                 labels={mcol: mname, 'country': ''})
    fig.update_layout(height=max(400, top_n * 22), coloraxis_showscale=False)
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Full Rankings Table</div>", unsafe_allow_html=True)
    top.columns = ['Country', mname]
    st.dataframe(top, use_container_width=True, height=400)

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:32px 0 16px 0; border-top:1px solid #1e2a45; margin-top:40px;'>
    <span style='font-size:11px; letter-spacing:0.15em; color:#334155;'>
        NEXUS ENERGY INTELLIGENCE · DATA: OUR WORLD IN DATA · BUILT WITH STREAMLIT
    </span>
</div>
""", unsafe_allow_html=True)
