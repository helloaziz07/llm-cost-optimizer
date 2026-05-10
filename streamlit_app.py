"""
Streamlit Dashboard for LLM Cost Optimizer
Clean, production-ready monitoring interface.
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="LLM Cost Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Design System — Professional, Clean CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* KPI Card Styling */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Accent colors for KPI values */
    .kpi-blue .kpi-value { color: #3b82f6; }
    .kpi-emerald .kpi-value { color: #10b981; }
    .kpi-amber .kpi-value { color: #f59e0b; }
    .kpi-violet .kpi-value { color: #8b5cf6; }

    /* Section Title */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Result card */
    .result-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
    }

    /* Status badges */
    .badge-hit {
        display: inline-block;
        background: #ecfdf5;
        color: #059669;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid #a7f3d0;
    }
    .badge-miss {
        display: inline-block;
        background: #fef2f2;
        color: #dc2626;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid #fecaca;
    }
    .badge-online {
        display: inline-block;
        background: #ecfdf5;
        color: #059669;
        font-weight: 500;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        border: 1px solid #a7f3d0;
    }

    /* Stat row inside result */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.85rem;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-label { color: #64748b; font-weight: 500; }
    .stat-value { color: #1e293b; font-weight: 600; }

    /* Override Streamlit metric cards to be invisible */
    [data-testid="stMetric"] {
        background: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    /* Clean divider */
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 32px 0;
        border: none;
    }

    /* Dashboard header */
    .dash-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }
    .dash-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .dash-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0 0 24px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        padding: 24px 0 8px;
        border-top: 1px solid #e2e8f0;
        margin-top: 40px;
    }

    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        .kpi-card {
            background: #1e293b;
            border-color: #334155;
        }
        .kpi-value { color: #f1f5f9 !important; }
        .kpi-label { color: #94a3b8; }
        .kpi-sub { color: #64748b; }
        .kpi-blue .kpi-value { color: #60a5fa !important; }
        .kpi-emerald .kpi-value { color: #34d399 !important; }
        .kpi-amber .kpi-value { color: #fbbf24 !important; }
        .kpi-violet .kpi-value { color: #a78bfa !important; }
        .section-title { color: #f1f5f9; border-color: #334155; }
        .result-card { background: #1e293b; border-color: #334155; }
        .stat-row { border-color: #334155; }
        .stat-label { color: #94a3b8; }
        .stat-value { color: #e2e8f0; }
        .dash-title { color: #f1f5f9; }
        .divider { background: #334155; }
        .footer { border-color: #334155; }
    }

    /* Streamlit dark mode class overrides */
    [data-theme="dark"] .kpi-card,
    .stApp[data-theme="dark"] .kpi-card {
        background: #1e293b;
        border-color: #334155;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================
def fetch(endpoint: str):
    """Fetch data from API"""
    try:
        r = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def send_query(query_text: str, max_tokens: int = 500, temperature: float = 0.7):
    """Send query to backend pipeline"""
    try:
        r = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query_text, "max_tokens": max_tokens, "temperature": temperature},
            timeout=30
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def kpi_card(label: str, value: str, sub: str = "", accent: str = ""):
    """Render a clean KPI card"""
    cls = f"kpi-card kpi-{accent}" if accent else "kpi-card"
    html = f"""
    <div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# Sidebar
# =============================================================================
API_BASE_URL = st.sidebar.text_input("Backend URL", "http://localhost:8000")

st.sidebar.markdown("---")

if st.sidebar.button("↻  Refresh", use_container_width=True):
    st.rerun()

auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)")
if auto_refresh:
    time.sleep(5)
    st.rerun()

st.sidebar.markdown("---")

if st.sidebar.button("Clear Cache", use_container_width=True):
    try:
        requests.post(f"{API_BASE_URL}/cache/clear", timeout=5)
        st.sidebar.success("Cache cleared")
        time.sleep(1)
        st.rerun()
    except Exception:
        st.sidebar.error("Failed")

if st.sidebar.button("Reset All Data", use_container_width=True):
    try:
        requests.post(f"{API_BASE_URL}/clear-all", timeout=5)
        st.sidebar.success("All data cleared")
        time.sleep(1)
        st.rerun()
    except Exception:
        st.sidebar.error("Failed")

st.sidebar.markdown("---")

# System config in sidebar
config_data_sb = fetch("/config")
if config_data_sb:
    with st.sidebar.expander("System Configuration"):
        st.json(config_data_sb)


# =============================================================================
# Fetch Data
# =============================================================================
metrics_data = fetch("/metrics")
recent_queries = fetch("/recent-queries?limit=20")

# =============================================================================
# Header
# =============================================================================
st.markdown("""
<div class="dash-header">
    <p class="dash-title">⚡ LLM Cost Optimizer</p>
    <span class="badge-online">Online</span>
</div>
<p class="dash-subtitle">Prompt Optimization → Semantic Caching → Model Selection → LLM Execution</p>
""", unsafe_allow_html=True)

# Connection check
if metrics_data is None:
    st.error("Cannot connect to backend. Start the server with: `python -m uvicorn main:app --port 8000`")
    st.stop()

# Extract metrics
cache_m = metrics_data.get('cache', {})
tracking_m = metrics_data.get('tracking', {})


# =============================================================================
# SECTION 1: KPI Cards
# =============================================================================
c1, c2, c3, c4 = st.columns(4)

total_queries = tracking_m.get('total_queries', 0)
hit_rate = cache_m.get('hit_rate', 0) * 100
total_cost = cache_m.get('total_cost', 0)
cost_saved = cache_m.get('total_cost_saved', 0)
cost_reduction = cache_m.get('cost_reduction_percent', 0)
cache_hits = cache_m.get('cache_hits', 0)
cache_misses = cache_m.get('cache_misses', 0)

with c1:
    kpi_card("Total Queries", str(total_queries), f"{cache_hits} hits · {cache_misses} misses", "blue")

with c2:
    kpi_card("Cache Hit Rate", f"{hit_rate:.1f}%", f"{cost_reduction:.1f}% cost reduction", "emerald")

with c3:
    kpi_card("Total Cost", f"${total_cost:.4f}", f"{cache_m.get('llm_tokens_used', 0)} tokens used", "amber")

with c4:
    kpi_card("Cost Saved", f"${cost_saved:.4f}", f"{cache_m.get('llm_tokens_saved', 0)} tokens saved", "violet")


# =============================================================================
# SECTION 2: Test Query Panel
# =============================================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Test Query</p>', unsafe_allow_html=True)

if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = None

with st.form("query_form"):
    query_input = st.text_area(
        "Enter your query",
        placeholder="e.g. Explain the concept of machine learning in simple terms...",
        height=80,
        label_visibility="collapsed"
    )

    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        max_tokens = st.slider("Max Tokens", 50, 1000, 500)
    with fc2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Send Query", use_container_width=True)

if submitted and query_input:
    with st.spinner("Processing..."):
        result = send_query(query_input, max_tokens, temperature)
        if result:
            st.session_state.last_result = result
            st.session_state.last_query = query_input

# Display result
if st.session_state.last_result:
    res = st.session_state.last_result
    cached = res.get('cached', False)

    col_resp, col_stats = st.columns([3, 1])

    with col_resp:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(res.get('response', ''))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        badge = '<span class="badge-hit">Cache Hit</span>' if cached else '<span class="badge-miss">Cache Miss</span>'
        st.markdown(badge, unsafe_allow_html=True)

        stats_html = '<div style="margin-top: 12px;">'

        if cached and res.get('similarity_score'):
            stats_html += f"""
            <div class="stat-row"><span class="stat-label">Similarity</span><span class="stat-value">{res['similarity_score']:.4f}</span></div>
            """
        if res.get('selected_model'):
            stats_html += f"""
            <div class="stat-row"><span class="stat-label">Model</span><span class="stat-value">{res['selected_model']}</span></div>
            """

        stats_html += f"""
        <div class="stat-row"><span class="stat-label">Tokens Used</span><span class="stat-value">{res.get('tokens_used', 0)}</span></div>
        <div class="stat-row"><span class="stat-label">Tokens Saved</span><span class="stat-value">{res.get('tokens_saved', 0)}</span></div>
        <div class="stat-row"><span class="stat-label">Cost</span><span class="stat-value">${res.get('cost', 0):.6f}</span></div>
        <div class="stat-row"><span class="stat-label">Saved</span><span class="stat-value">${res.get('cost_saved', 0):.6f}</span></div>
        <div class="stat-row"><span class="stat-label">Latency</span><span class="stat-value">{res.get('latency_ms', 0):.1f}ms</span></div>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)

    if st.button("Clear Result", key="clear_result"):
        st.session_state.last_result = None
        st.session_state.last_query = None
        st.rerun()


# =============================================================================
# SECTION 3: Cost & Token Analytics
# =============================================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Analytics</p>', unsafe_allow_html=True)

chart_c1, chart_c2 = st.columns(2)

palette = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

with chart_c1:
    cost_actual = cache_m.get('total_cost', 0)
    cost_saved_val = cache_m.get('total_cost_saved', 0)

    if cost_actual + cost_saved_val > 0:
        fig = go.Figure(data=[go.Pie(
            labels=['Actual Cost', 'Cost Saved'],
            values=[cost_actual, cost_saved_val],
            hole=0.55,
            marker=dict(colors=[palette[2], palette[1]]),
            textinfo='label+percent',
            textfont=dict(size=13, family='Inter'),
            hovertemplate='%{label}: $%{value:.6f}<extra></extra>'
        )])
        fig.update_layout(
            title=dict(text='Cost Breakdown', font=dict(size=15, family='Inter', color='#475569')),
            showlegend=False,
            height=300,
            margin=dict(t=50, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cost data yet — send some queries first")

with chart_c2:
    tokens_used = cache_m.get('llm_tokens_used', 0)
    tokens_saved = cache_m.get('llm_tokens_saved', 0)

    if tokens_used + tokens_saved > 0:
        fig = go.Figure(data=[go.Pie(
            labels=['Tokens Used', 'Tokens Saved'],
            values=[tokens_used, tokens_saved],
            hole=0.55,
            marker=dict(colors=[palette[0], palette[3]]),
            textinfo='label+percent',
            textfont=dict(size=13, family='Inter'),
            hovertemplate='%{label}: %{value:,}<extra></extra>'
        )])
        fig.update_layout(
            title=dict(text='Token Usage', font=dict(size=15, family='Inter', color='#475569')),
            showlegend=False,
            height=300,
            margin=dict(t=50, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No token data yet — send some queries first")


# =============================================================================
# SECTION 4: Recent Queries Table
# =============================================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Recent Queries</p>', unsafe_allow_html=True)

if recent_queries and recent_queries.get('queries'):
    queries = recent_queries['queries']
    df = pd.DataFrame(queries)

    if not df.empty:
        # Select columns
        cols = ['query_id', 'original_prompt', 'cache_hit', 'selected_model',
                'llm_cost', 'cost_saved', 'total_time_ms', 'status']
        available = [c for c in cols if c in df.columns]
        display = df[available].copy()

        # Rename for clean headers
        rename_map = {
            'query_id': 'ID',
            'original_prompt': 'Query',
            'cache_hit': 'Cached',
            'selected_model': 'Model',
            'llm_cost': 'Cost',
            'cost_saved': 'Saved',
            'total_time_ms': 'Latency (ms)',
            'status': 'Status'
        }
        display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns}, inplace=True)

        # Format cached column as text badges
        if 'Cached' in display.columns:
            display['Cached'] = display['Cached'].apply(lambda x: '✓ Hit' if x else '✗ Miss')

        # Format latency
        if 'Latency (ms)' in display.columns:
            display['Latency (ms)'] = display['Latency (ms)'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")

        st.dataframe(
            display,
            use_container_width=True,
            height=min(400, 50 + len(display) * 35),
            hide_index=True,
            column_config={
                "ID": st.column_config.TextColumn(width="small"),
                "Query": st.column_config.TextColumn(width="large"),
                "Cached": st.column_config.TextColumn(width="small"),
                "Model": st.column_config.TextColumn(width="medium"),
                "Cost": st.column_config.TextColumn(width="small"),
                "Saved": st.column_config.TextColumn(width="small"),
                "Latency (ms)": st.column_config.TextColumn(width="small"),
                "Status": st.column_config.TextColumn(width="small"),
            }
        )
else:
    st.info("No queries yet — use the test panel above to send your first query")


# =============================================================================
# Footer
# =============================================================================
st.markdown("""
<div class="footer">
    LLM Cost Optimizer v1.0 · Prompt Optimization · Semantic Caching · Model Selection · LLM Execution
</div>
""", unsafe_allow_html=True)
