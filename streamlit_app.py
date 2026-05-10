"""
Streamlit Dashboard — LLM Cost Optimizer
A clean, native, professional interface without CSS hacks.
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="LLM Cost Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS to hide default header/footer and improve padding slightly
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================
def fetch(endpoint: str):
    try:
        r = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def send_query(query_text: str, max_tokens: int = 500, temperature: float = 0.7):
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


# =============================================================================
# Sidebar Structure
# =============================================================================
with st.sidebar:
    st.header("Connection")
    API_BASE_URL = st.text_input("Backend URL", "http://localhost:8000", label_visibility="collapsed")
    
    st.divider()
    
    st.header("Controls")
    if st.button("↻ Refresh Dashboard", use_container_width=True):
        st.rerun()
        
    auto_refresh = st.checkbox("Auto-refresh (5s)")
    if auto_refresh:
        time.sleep(5)
        st.rerun()
        
    st.divider()
    
    st.header("Maintenance")
    if st.button("Clear Semantic Cache", use_container_width=True):
        try:
            requests.post(f"{API_BASE_URL}/cache/clear", timeout=5)
            st.success("Cache cleared")
            time.sleep(1)
            st.rerun()
        except:
            st.error("Operation failed")

    if st.button("Reset All Metrics", use_container_width=True):
        try:
            requests.post(f"{API_BASE_URL}/clear-all", timeout=5)
            st.success("System reset")
            time.sleep(1)
            st.rerun()
        except:
            st.error("Operation failed")

    st.divider()
    cfg = fetch("/config")
    if cfg:
        with st.expander("System Configuration"):
            st.json(cfg)


# =============================================================================
# Data Fetching
# =============================================================================
metrics_data = fetch("/metrics")
recent_queries = fetch("/recent-queries?limit=20")
batch_stats = fetch("/batching/stats")
batch_details = fetch("/batching/details")
cache_stats = fetch("/cache/stats")

# =============================================================================
# Main Layout - Header
# =============================================================================
col_title, col_status = st.columns([5, 1])
with col_title:
    st.title("⚡ LLM Cost Optimizer")
    st.markdown("Intelligent Middleware for Prompt Optimization, Routing & Caching")
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("🟢 System Online")

if metrics_data is None:
    st.error("Cannot connect to backend. Please ensure the FastAPI server is running: `python -m uvicorn main:app --port 8000`")
    st.stop()

st.divider()

cache_m = metrics_data.get('cache', {})
tracking_m = metrics_data.get('tracking', {})

# =============================================================================
# Key Performance Indicators (Native)
# =============================================================================
tq = tracking_m.get('total_queries', 0)
hits = cache_m.get('cache_hits', 0)
hit_rate = cache_m.get('hit_rate', 0) * 100
cost = cache_m.get('total_cost', 0)
saved = cache_m.get('total_cost_saved', 0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Queries", f"{tq:,}", f"{hits} Cache Hits", delta_color="normal")
m2.metric("Hit Rate", f"{hit_rate:.1f}%", f"{cache_m.get('cost_reduction_percent', 0):.1f}% Cost Drop", delta_color="normal")
m3.metric("Total Execution Cost", f"${cost:.4f}", f"{cache_m.get('llm_tokens_used', 0):,} Tokens Used", delta_color="off")
m4.metric("Total Savings", f"${saved:.4f}", f"{cache_m.get('llm_tokens_saved', 0):,} Tokens Saved", delta_color="normal")

st.divider()

# =============================================================================
# Interactive Query Sandbox
# =============================================================================
st.subheader("🧪 Query Sandbox")

if 'last_result' not in st.session_state: st.session_state.last_result = None
if 'last_query' not in st.session_state: st.session_state.last_query = None

with st.container(border=True):
    with st.form("query_form"):
        qi = st.text_area("Input Prompt", placeholder="Type a prompt to test the optimization and routing pipeline...", height=100)
        
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            mt = st.slider("Max Tokens", 50, 1000, 500)
        with c2:
            temp = st.slider("Temperature", 0.0, 1.0, 0.7)
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Execute Pipeline", use_container_width=True)

if submitted and qi:
    with st.spinner("Processing through Optimizer Pipeline..."):
        r = send_query(qi, mt, temp)
        if r:
            st.session_state.last_result = r
            st.session_state.last_query = qi

if st.session_state.last_result:
    res = st.session_state.last_result
    cached = res.get('cached', False)
    
    st.markdown("### Result")
    res_col, stat_col = st.columns([7, 3])
    
    with res_col:
        st.info(res.get("response", ""))
        
    with stat_col:
        if cached:
            st.success("⚡ Cache Hit")
        else:
            st.warning("⚙️ Live LLM Generation")
            
        with st.container(border=True):
            if res.get('selected_model'):
                st.write(f"**Model:** {res['selected_model']}")
            if cached and res.get('similarity_score'):
                st.write(f"**Similarity:** {res['similarity_score']:.3f}")
            
            st.write(f"**Latency:** {float(res.get('latency_ms', 0)):.0f} ms")
            st.write(f"**Tokens Used:** {res.get('tokens_used', 0)}")
            st.write(f"**Cost:** ${float(res.get('cost', 0)):.6f}")
            st.write(f"**Saved:** ${float(res.get('cost_saved', 0)):.6f}")
        
        if st.button("Dismiss Result", use_container_width=True):
            st.session_state.last_result = None
            st.session_state.last_query = None
            st.rerun()

st.divider()

# =============================================================================
# Visual Analytics
# =============================================================================
st.subheader("📈 System Analytics")

chart_col1, chart_col2 = st.columns(2)
colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

with chart_col1:
    cost_actual = cache_m.get('total_cost', 0)
    cost_saved_val = cache_m.get('total_cost_saved', 0)
    
    if cost_actual + cost_saved_val > 0:
        with st.container(border=True):
            fig1 = go.Figure(data=[go.Pie(
                labels=['Actual Spend', 'Retained Savings'],
                values=[cost_actual, cost_saved_val],
                hole=0.6,
                marker=dict(colors=[colors[2], colors[1]]),
                textinfo='label+percent'
            )])
            fig1.update_layout(title='Cost Efficiency Breakdown', height=300, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Insufficient data for cost breakdown.")

with chart_col2:
    tokens_used = cache_m.get('llm_tokens_used', 0)
    tokens_saved = cache_m.get('llm_tokens_saved', 0)
    
    if tokens_used + tokens_saved > 0:
        with st.container(border=True):
            fig2 = go.Figure(data=[go.Pie(
                labels=['Tokens Sent to LLM', 'Tokens Prevented'],
                values=[tokens_used, tokens_saved],
                hole=0.6,
                marker=dict(colors=[colors[0], colors[3]]),
                textinfo='label+percent'
            )])
            fig2.update_layout(title='Token Utilization', height=300, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Insufficient data for token analysis.")

st.divider()

# =============================================================================
# Routing & Batching Insights
# =============================================================================
st.subheader("🔄 Routing & Batching Engine")

if batch_stats and batch_stats.get('batches_by_model'):
    model_data = batch_stats['batches_by_model']
    
    with st.container(border=True):
        bar_colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#ef4444', '#14b8a6']
        fig_bar = go.Figure(data=[go.Bar(
            x=list(model_data.keys()), 
            y=list(model_data.values()),
            marker_color=[bar_colors[i % len(bar_colors)] for i in range(len(model_data.keys()))],
            text=list(model_data.values()), 
            textposition='auto'
        )])
        fig_bar.update_layout(
            title='Batches Created per Model',
            height=300, margin=dict(t=40, b=20, l=20, r=20),
            xaxis_title="Model", yaxis_title="Number of Batches"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

if batch_details and batch_details.get('batches'):
    st.markdown("#### Active Batch Ledger")
    
    # We will display the batches natively in an expander for clean UI
    for batch in batch_details['batches']:
        with st.expander(f"📦 **{batch['batch_id']}** — Model: **{batch['model']}** | Queries: **{batch['size']}** | Tokens: **{batch['total_tokens']}** | Trigger: **{batch['close_reason']}**"):
            if batch.get('queries'):
                q_df = pd.DataFrame(batch['queries'])
                st.dataframe(q_df, use_container_width=True, hide_index=True)
else:
    if not batch_stats:
        st.info("No batching data available. Run the load test simulator to generate traffic.")

st.divider()

# =============================================================================
# Cache & Query Ledger
# =============================================================================
st.subheader("🗄️ Data Ledgers")

tab1, tab2 = st.tabs(["Top Semantic Hits", "Recent Query Log"])

with tab1:
    if cache_stats and cache_stats.get('stats', {}).get('top_queries'):
        tq_data = cache_stats['stats']['top_queries']
        df_cache = pd.DataFrame(tq_data)
        
        df_cache = df_cache.rename(columns={
            'query': 'Semantic Root Query',
            'hits': 'Hit Count',
            'tokens_saved': 'Tokens Saved',
            'avg_similarity': 'Avg. Match Confidence'
        })
        df_cache['Avg. Match Confidence'] = df_cache['Avg. Match Confidence'].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(df_cache, use_container_width=True, hide_index=True)
    else:
        st.info("Cache is currently empty or has no hits.")

with tab2:
    if recent_queries and recent_queries.get('queries'):
        df_queries = pd.DataFrame(recent_queries['queries'])
        if not df_queries.empty:
            display_cols = ['query_id', 'original_prompt', 'cache_hit', 'selected_model', 'llm_cost', 'total_time_ms']
            df_display = df_queries[[c for c in display_cols if c in df_queries.columns]].copy()
            
            df_display = df_display.rename(columns={
                'query_id': 'Trace ID',
                'original_prompt': 'Original Request',
                'cache_hit': 'Resolution',
                'selected_model': 'Routed Model',
                'llm_cost': 'Execution Cost',
                'total_time_ms': 'E2E Latency'
            })
            
            if 'Resolution' in df_display.columns:
                df_display['Resolution'] = df_display['Resolution'].apply(lambda x: '⚡ Cache' if x else '🧠 Generate')
            if 'Execution Cost' in df_display.columns:
                df_display['Execution Cost'] = pd.to_numeric(df_display['Execution Cost'], errors='coerce').apply(lambda x: f"${x:.5f}" if pd.notna(x) else "-")
            if 'E2E Latency' in df_display.columns:
                df_display['E2E Latency'] = df_display['E2E Latency'].apply(lambda x: f"{x:.0f} ms" if pd.notna(x) else "-")
                
            st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No query logs available.")
