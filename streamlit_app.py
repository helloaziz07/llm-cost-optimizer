"""
Streamlit Dashboard for LLM Cost Optimizer
Comprehensive monitoring and query interface.
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="LLM Cost Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_BASE_URL = st.sidebar.text_input("Backend URL", "http://localhost:8000")

# Custom CSS for premium look
st.markdown("""
<style>
    /* Gradient metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    [data-testid="stMetric"] label {
        color: #ffffff !important;
        font-weight: 600;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: bold;
    }

    /* Different gradient colors */
    [data-testid="column"]:nth-child(1) [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="column"]:nth-child(2) [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    [data-testid="column"]:nth-child(3) [data-testid="stMetric"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    [data-testid="column"]:nth-child(4) [data-testid="stMetric"] {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    [data-testid="column"]:nth-child(5) [data-testid="stMetric"] {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    [data-testid="column"]:nth-child(6) [data-testid="stMetric"] {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
    }

    /* Section headers */
    h1, h2, h3 {
        color: #1f1f1f;
        font-weight: 700;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def fetch_data(endpoint):
    """Fetch data from API endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def send_query(query_text, max_tokens=500, temperature=0.7):
    """Send query to backend"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": query_text,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# Sidebar
st.sidebar.title("💰 Cost Optimizer")
st.sidebar.markdown("---")

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
if auto_refresh:
    time.sleep(5)
    st.rerun()

# Manual refresh
if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.rerun()

# Clear cache
if st.sidebar.button("🗑️ Clear Cache", use_container_width=True):
    try:
        requests.post(f"{API_BASE_URL}/cache/clear", timeout=5)
        st.sidebar.success("Cache cleared!")
        time.sleep(1)
        st.rerun()
    except Exception:
        st.sidebar.error("Failed to clear cache")

st.sidebar.markdown("---")

# Fetch data
metrics_data = fetch_data("/metrics")
recent_queries = fetch_data("/recent-queries?limit=20")

# Main title
st.title("🎯 LLM Cost Optimizer Dashboard")
st.markdown("*Prompt Optimization → Semantic Caching → Model Selection → Batching → LLM*")
st.markdown("---")

# Check backend connection
if metrics_data is None:
    st.error("❌ Cannot connect to backend. Make sure the server is running!")
    st.code("uvicorn main:app --reload --port 8000")
    st.stop()

# Extract metrics
cache_metrics = metrics_data.get('cache', {})
tracking_metrics = metrics_data.get('tracking', {})
batching_metrics = metrics_data.get('batching', {})
config_data = metrics_data.get('config', {})

# =============================================================================
# SECTION 1: KEY METRICS
# =============================================================================
st.header("📊 Real-Time System Metrics")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Queries", tracking_metrics.get('total_queries', 0))

with col2:
    hit_rate = cache_metrics.get('hit_rate', 0) * 100
    st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")

with col3:
    st.metric("Cache Hits", cache_metrics.get('cache_hits', 0))

with col4:
    st.metric("Cache Misses", cache_metrics.get('cache_misses', 0))

with col5:
    total_cost = cache_metrics.get('total_cost', 0)
    st.metric("Total Cost", f"${total_cost:.4f}")

with col6:
    cost_saved = cache_metrics.get('total_cost_saved', 0)
    st.metric("Cost Saved", f"${cost_saved:.4f}")

# =============================================================================
# SECTION 2: QUERY INPUT PANEL
# =============================================================================
st.markdown("---")
st.header("🧪 Test Query Panel")

# Initialize session state for query result
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = None

with st.form("query_form"):
    query_input = st.text_area(
        "Enter your query:",
        placeholder="Example: Explain the concept of machine learning in simple terms...",
        height=100
    )

    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider("Max Tokens", 50, 1000, 500)
    with col2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    col_submit, col_clear = st.columns(2)
    with col_submit:
        submit_button = st.form_submit_button("🚀 Send Query", use_container_width=True)
    with col_clear:
        clear_button = st.form_submit_button("🗑️ Clear Result", use_container_width=True)

if submit_button and query_input:
    with st.spinner("Processing query through pipeline..."):
        result = send_query(query_input, max_tokens, temperature)
        if result:
            st.session_state.last_result = result
            st.session_state.last_query = query_input

if clear_button:
    st.session_state.last_result = None
    st.session_state.last_query = None
    st.rerun()

# Display result if available
if st.session_state.last_result:
    result = st.session_state.last_result

    display_query = st.session_state.last_query
    if len(display_query) > 80:
        display_query = display_query[:80] + "..."
    st.subheader(f"📝 Query: {display_query}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Response:**")
        st.write(result.get('response', ''))

    with col2:
        st.markdown("**📊 Query Metrics**")

        cached = result.get('cached', False)
        if cached:
            st.success("✅ CACHE HIT")
            st.write(f"**Similarity:** {result.get('similarity_score', 0):.4f}")
        else:
            st.error("❌ CACHE MISS")
            st.write(f"**Model:** {result.get('selected_model', 'N/A')}")

        st.write(f"**Tokens Used:** {result.get('tokens_used', 0)}")
        st.write(f"**Tokens Saved:** {result.get('tokens_saved', 0)}")
        st.write(f"**Cost:** ${result.get('cost', 0):.6f}")
        st.write(f"**Cost Saved:** ${result.get('cost_saved', 0):.6f}")
        st.write(f"**Latency:** {result.get('latency_ms', 0):.2f}ms")

# =============================================================================
# SECTION 3: RECENT 20 QUERIES
# =============================================================================
st.markdown("---")
st.header("📜 Recent 20 Queries — Complete Tracking")

if recent_queries and recent_queries.get('queries'):
    queries = recent_queries['queries']

    # Create DataFrame for display
    df_queries = pd.DataFrame(queries)

    if not df_queries.empty:
        # Select and rename columns for display
        display_cols = [
            'query_id', 'timestamp', 'original_prompt', 'cache_hit',
            'cache_similarity', 'selected_model', 'llm_tokens',
            'llm_cost', 'cost_saved', 'total_time_ms', 'status'
        ]

        available_cols = [c for c in display_cols if c in df_queries.columns]
        df_display = df_queries[available_cols].copy()

        # Style the dataframe
        def highlight_cache(row):
            if row.get('cache_hit', False):
                return ['background-color: #06d638'] * len(row)
            else:
                return ['background-color: #ed071c'] * len(row)

        styled_df = df_display.style.apply(highlight_cache, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)

        cache_hits = df_display[df_display['cache_hit'] == True].shape[0] if 'cache_hit' in df_display.columns else 0
        cache_misses = df_display[df_display['cache_hit'] == False].shape[0] if 'cache_hit' in df_display.columns else 0

        with col1:
            st.metric("Recent Hits", cache_hits)
        with col2:
            st.metric("Recent Misses", cache_misses)
        with col3:
            avg_time = df_display['total_time_ms'].mean() if 'total_time_ms' in df_display.columns else 0
            st.metric("Avg Response Time", f"{avg_time:.0f}ms")
        with col4:
            st.metric("Queries Shown", len(df_display))
else:
    st.info("No queries yet. Send some queries using the test panel above!")

# =============================================================================
# SECTION 4: CACHE DETAILS
# =============================================================================
st.markdown("---")
st.header("🗄️ Semantic Cache Details")

cache_stats = fetch_data("/cache/stats")
cache_entries = fetch_data("/cache/entries?limit=20")

if cache_stats:
    col1, col2, col3 = st.columns(3)

    stats = cache_stats.get('stats', {})

    with col1:
        st.metric("Cache Entries", stats.get('total_entries', 0))
        st.metric("Avg Hits/Entry", stats.get('avg_hits_per_entry', 0))

    with col2:
        st.metric("Tokens Saved", cache_metrics.get('llm_tokens_saved', 0))
        st.metric("Total Evictions", cache_metrics.get('evictions', 0))

    with col3:
        thresholds = cache_stats.get('thresholds', {})
        st.write("**Adaptive Thresholds:**")
        st.write(f"• Short queries: {thresholds.get('short', 0.85):.2f}")
        st.write(f"• Medium queries: {thresholds.get('medium', 0.80):.2f}")
        st.write(f"• Long queries: {thresholds.get('long', 0.75):.2f}")

# Cache entries table
if cache_entries and cache_entries.get('entries'):
    st.subheader("📋 Cache Entries")
    entries_df = pd.DataFrame(cache_entries['entries'])
    st.dataframe(entries_df, use_container_width=True)

# =============================================================================
# SECTION 5: CACHE EVICTION TRACKING
# =============================================================================
st.markdown("---")
st.header("🗑️ Cache Eviction Tracking")

eviction_data = fetch_data("/cache/evictions?limit=10")

col1, col2 = st.columns([1, 3])

with col1:
    total_evictions = cache_metrics.get('evictions', 0)
    st.metric("Total Evictions", total_evictions)

    cache_size = cache_metrics.get('cache_size', 0)
    st.metric("Current Cache Size", cache_size)

    if total_evictions > 0:
        st.info(f"💡 {total_evictions} entries removed to optimize cache performance")

with col2:
    if eviction_data and eviction_data.get('evictions'):
        evictions = eviction_data['evictions']

        st.subheader("📜 Recent 10 Evicted Queries")

        eviction_records = []
        for i, ev in enumerate(evictions[-10:], 1):
            eviction_records.append({
                "#": i,
                "Query": ev.get('query', '')[:60] + "..." if len(ev.get('query', '')) > 60 else ev.get('query', ''),
                "Hits": ev.get('hits', 0),
                "Age (hrs)": ev.get('age_hours', 0),
                "Reason": ev.get('reason', 'Low value score'),
                "Evicted At": ev.get('timestamp', 'N/A')[:19] if ev.get('timestamp') else 'N/A'
            })

        if eviction_records:
            eviction_df = pd.DataFrame(eviction_records)

            def style_eviction(row):
                reason = row.get('Reason', '')
                if 'Low value' in reason:
                    return ['background-color: #fac92d'] * len(row)
                elif 'TTL' in reason or 'expired' in reason.lower():
                    return ['background-color: #f0293b'] * len(row)
                else:
                    return ['background-color: #316feb'] * len(row)

            styled_eviction = eviction_df.style.apply(style_eviction, axis=1)
            st.dataframe(styled_eviction, use_container_width=True, height=350)

            st.markdown("**Eviction Reasons Legend:**")
            st.markdown("- 🟡 **Low value score**: Entry had few hits, old age, or low similarity matches")
            st.markdown("- 🔴 **TTL expired**: Entry exceeded time-to-live limit")
            st.markdown("- ⚪ **Other**: Manual eviction or cache clear")
    else:
        st.info("No evictions yet. Cache will evict entries when it reaches maximum capacity.")

# =============================================================================
# SECTION 6: BATCHING STATS
# =============================================================================
st.markdown("---")
st.header("📦 Batching Statistics")

batching_stats = fetch_data("/batching/stats")

if batching_stats:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Batches", batching_stats.get('total_batches_created', 0))
    with col2:
        st.metric("Requests Batched", batching_stats.get('total_requests_batched', 0))
    with col3:
        st.metric("Avg Batch Size", batching_stats.get('avg_batch_size', 0))

    # Batches by model
    batches_by_model = batching_stats.get('batches_by_model', {})
    if batches_by_model:
        st.subheader("Batches by Model")
        model_df = pd.DataFrame([
            {"Model": k, "Batches": v}
            for k, v in batches_by_model.items()
        ])

        fig = px.bar(model_df, x='Model', y='Batches',
                     title='Batch Distribution by Model',
                     color='Batches',
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SECTION 7: COST ANALYTICS
# =============================================================================
st.markdown("---")
st.header("💰 Cost Analytics")

col1, col2 = st.columns(2)

with col1:
    cost_data = {
        'Category': ['Actual Cost', 'Cost Saved'],
        'Amount': [
            cache_metrics.get('total_cost', 0),
            cache_metrics.get('total_cost_saved', 0)
        ]
    }

    if sum(cost_data['Amount']) > 0:
        fig = px.pie(
            values=cost_data['Amount'],
            names=cost_data['Category'],
            title='Cost Distribution',
            color_discrete_sequence=['#ff6384', '#36a2eb']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cost data yet")

with col2:
    token_data = {
        'Category': ['Tokens Used', 'Tokens Saved'],
        'Count': [
            cache_metrics.get('llm_tokens_used', 0),
            cache_metrics.get('llm_tokens_saved', 0)
        ]
    }

    if sum(token_data['Count']) > 0:
        fig = px.pie(
            values=token_data['Count'],
            names=token_data['Category'],
            title='Token Usage Distribution',
            color_discrete_sequence=['#ff9f40', '#4bc0c0']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No token data yet")

# =============================================================================
# SECTION 8: SYSTEM CONFIGURATION
# =============================================================================
st.markdown("---")
st.header("⚙️ System Configuration")

with st.expander("View Current Configuration"):
    st.json(config_data)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💰 LLM Cost Optimizer v1.0.0</p>
    <p>Prompt Optimization • Semantic Caching • Model Selection • Request Batching</p>
</div>
""", unsafe_allow_html=True)
