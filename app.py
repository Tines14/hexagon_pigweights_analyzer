import streamlit as st

st.set_page_config(
    page_title="Pig Weight Analyzer",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');

html, body, p, h1, h2, h3, h4, h5, h6 {
    font-family: 'Prompt', 'Segoe UI', sans-serif !important;
}

            
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-right: 2px solid #e94560;
}
[data-testid="stSidebar"] * {
    font-family: 'Prompt', sans-serif !important;
    color: #f0f0f0 !important;
}

/* Sidebar nav button */
.nav-btn {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    padding: 14px 20px;
    margin: 6px 0;
    border-radius: 12px;
    border: none;
    background: transparent;
    color: #c8c8d8 !important;
    font-family: 'Prompt', sans-serif !important;;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.25s ease;
    text-align: left;
}
.nav-btn:hover {
    background: rgba(233,69,96,0.15);
    color: #ffffff !important;
    transform: translateX(4px);
}
.nav-btn.active {
    background: linear-gradient(90deg, #e94560, #c0392b);
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(233,69,96,0.4);
}
            
/* Page header */
.page-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-left: 5px solid #e94560;
    border-radius: 12px;
    padding: 24px 30px;
    margin-bottom: 28px;
    color: white;
}
.page-header h1 { margin: 0; font-size: 26px; font-weight: 700; }
.page-header p  { margin: 6px 0 0; color: #aaa; font-size: 14px; }

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 2px dashed #e94560 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    background: rgba(233,69,96,0.04) !important;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 20px 24px;
    color: white;
    margin-bottom: 16px;
}
.weight-badge {
    display: inline-block;
    background: linear-gradient(90deg, #e94560, #c0392b);
    color: white;
    font-size: 22px;
    font-weight: 700;
    padding: 6px 20px;
    border-radius: 30px;
    margin-top: 8px;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2a2a4a;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    color: white;
}
.metric-card .val { font-size: 28px; font-weight: 700; color: #e94560; }
.metric-card .lbl { font-size: 13px; color: #aaa; margin-top: 4px; }

/* List item */
.pig-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 12px 20px;
    margin-bottom: 8px;
    color: white;
}
.pig-row .pig-name { font-weight: 600; font-size: 15px; }
.pig-row .pig-wt { color: #e94560; font-weight: 700; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "analyze"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 20px 0 28px;'>
            <div style='font-size:60px;'>🐷</div>
            <div style='font-size:20px; font-weight:700; color:white; margin-top:8px;'>Pig weight analysis system</div>
            <div style='font-size:16px; color:#aaa; margin-top:4px;'>v1.0.0</div>
        </div>
    """, unsafe_allow_html=True)

    pages = [
        ("📷", "Analyze Pig Weight", "analyze"),
        ("📊", "About System",    "about"),
    ]

    for icon, label, key in pages:
        active = "active" if st.session_state.page == key else ""
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("<hr style='border-color:#2a2a4a; margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size:11px; color:#555; text-align:center;'>
            Model: best.pt + RandomForest<br>
            v1.0.0
        </div>
    """, unsafe_allow_html=True)

# ─── Page routing ──────────────────────────────────────────────────────────────
if st.session_state.page == "analyze":
    from pages_src.page_analyze import render
    render()
elif st.session_state.page == "about":
    from pages_src.page_about import render
    render()
