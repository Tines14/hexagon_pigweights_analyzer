"""Page 2 - เกี่ยวกับระบบ"""

import os
import streamlit as st

def _find_project_root():
    """หา root ของโปรเจกต์โดยมองหา app.py"""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(current, "app.py")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.getcwd()

def _get_model_path(filename):
    root = _find_project_root()
    candidates = [
        os.path.join(root, filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def render():
    st.markdown("""
        <div class="page-header">
            <h1>📊 About System</h1>
            <p>Model details and usage instructions</p>
        </div>
    """, unsafe_allow_html=True)

    # ─── สถานะโมเดล ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 Model status")

    col1, col2 = st.columns(2)

    with col1:
        pt_path = _get_model_path("best.pt")
        exists  = pt_path is not None
        size    = f"{os.path.getsize(pt_path)/1e6:.1f} MB" if exists else "—"
        status  = "🟢 Ready" if exists else "🔴 File not found"
        color   = "#34dc0e" if exists else "#e2200b"

        st.markdown(f"""
            <div class="result-card">
                <div style='font-size:16px; font-weight:600;'>🎯 YOLOv8 (best.pt)</div>
                <div style='margin-top:8px; font-size:12px; color:#aaa;'>
                    Used to detect and draw Bounding Box around pigs
                </div>
                <div style='margin-top:12px;'>
                    <span style='background:{color}22; color:{color};
                                 padding:4px 14px; border-radius:20px;
                                 font-size:12px; font-weight:500;'>
                        {status}
                    </span>
                    &nbsp;&nbsp;
                    <span style='color:#777; font-size:12px;'>size: {size}</span>
                </div>
                <div style='margin-top:8px; font-size:12px; color:#555;'>
                    path: best.pt
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        skp_path = _get_model_path("random_forest.pkl")
        exists2  = skp_path is not None
        size2    = f"{os.path.getsize(skp_path)/1e6:.1f} MB" if exists2 else "—"
        status2  = "🟢 Ready" if exists2 else "🔴 File not found"
        color2   = "#34dc0e" if exists2 else "#e2200b"

        st.markdown(f"""
            <div class="result-card">
                <div style='font-size:16px; font-weight:600;'>🌲 Random Forest (.pkl)</div>
                <div style='margin-top:8px; font-size:12px; color:#aaa;'>
                    Get features from YOLO and predict weight in kg.
                </div>
                <div style='margin-top:12px;'>
                    <span style='background:{color2}22; color:{color2};
                                 padding:4px 14px; border-radius:20px;
                                 font-size:12px; font-weight:500;'>
                        {status2}
                    </span>
                    &nbsp;&nbsp;
                    <span style='color:#777; font-size:12px;'>ขนาด: {size2}</span>
                </div>
                <div style='margin-top:8px; font-size:12px; color:#555;'>
                    path: random_forest.pkl
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ─── Pipeline ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Pipeline การทำงาน")

    steps = [
        ("1", "📂", "Receive Image",
        "Supports JPG/PNG/BMP/WEBP/TIFF and ZIP files containing images"),
        ("2", "🎯", "YOLO Detection",
        "Detects pig locations in images, creates bounding boxes with confidence scores"),
        ("3", "📐", "Feature Extraction",
        "Extracts features from bbox: width, height, area, aspect ratio, confidence"),
        ("4", "🌲", "Random Forest Predict",
        "Imports features into RandomForest to predict weight (kg)"),
        ("5", "📊", "Display & Export",
        "Displays before/after images, weight, total items, and downloads to Excel"),
    ]

    for num, icon, title, desc in steps:
        st.markdown(f"""
            <div style='display:flex; gap:16px; margin-bottom:12px;
                        background:#1a1a2e; border:1px solid #2a2a4a;
                        border-radius:12px; padding:16px 20px; color:white;'>
                <div style='min-width:36px; height:36px; border-radius:50%;
                             background:#e94560; display:flex; align-items:center;
                             justify-content:center; font-weight:700; font-size:15px;'>
                    {num}
                </div>
                <div>
                    <div style='font-weight:600; font-size:15px;'>{icon} {title}</div>
                    <div style='color:#aaa; font-size:13px; margin-top:4px;'>{desc}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ─── การติดตั้ง ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛠️ Installing and running the app.")

    st.markdown("**Place the model files in the same folder as `app.py`:**")
    st.code("""
pig_weight_app/
├── app.py
├── best.pt               ← YOLOv8 model file
├── random_forest.pkl     ← RandomForest model file
├── requirements.txt
└── pages_src/
    ├── __init__.py
    ├── page_analyze.py
    └── page_about.py
    """, language="text")

    st.markdown("**Install dependencies:**")
    st.code("""
pip install -r requirements.txt
    """, language="bash")

    st.markdown("**Running app:**")
    st.code("""
streamlit run app.py
    """, language="bash")

    # ─── Features ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✨ All features at a glance")

    features = [
        ("📁", "Upload multiple images at once or ZIP"),        
        ("🎯", "YOLO automatically detects bounding boxes"),
        ("🌲", "RandomForest predicts weight (kg)"),
        ("🖼️", "Show before/after images with layout"),
        ("📋", "Results list sorted by weight"),
        ("📥", "Download results as Excel (.xlsx)"),
        ("📊", "Summary of avg/max/min weight"),
        ("🔄", "Demo mode when no model is generated"),
    ]

    cols = st.columns(2)
    for i, (icon, text) in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
                <div style='background:#1a1a2e; border:1px solid #2a2a4a;
                             border-radius:10px; padding:12px 16px; margin-bottom:8px;
                             color:white; font-size:14px;'>
                    {icon} &nbsp; {text}
                </div>
            """, unsafe_allow_html=True)