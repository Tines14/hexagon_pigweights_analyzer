"""หน้าที่ 2 - เกี่ยวกับระบบ"""

import os
import streamlit as st

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render():
    st.markdown("""
        <div class="page-header">
            <h1>📊 เกี่ยวกับระบบ</h1>
            <p>รายละเอียดโมเดลและวิธีการใช้งาน</p>
        </div>
    """, unsafe_allow_html=True)

    # ─── สถานะโมเดล ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 สถานะโมเดล AI")

    col1, col2 = st.columns(2)

    with col1:
        pt_path = os.path.join(MODEL_DIR, "best.pt")
        exists  = os.path.exists(pt_path)
        size    = f"{os.path.getsize(pt_path)/1e6:.1f} MB" if exists else "—"
        status  = "🟢 พร้อมใช้งาน" if exists else "🔴 ไม่พบไฟล์"
        color   = "#2ecc71" if exists else "#e94560"

        st.markdown(f"""
            <div class="result-card">
                <div style='font-size:18px; font-weight:700;'>🎯 YOLOv8 (best.pt)</div>
                <div style='margin-top:10px; font-size:14px; color:#aaa;'>
                    ใช้ตรวจจับและวาด Bounding Box รอบตัวหมู
                </div>
                <div style='margin-top:14px;'>
                    <span style='background:{color}22; color:{color};
                                 padding:4px 14px; border-radius:20px;
                                 font-size:13px; font-weight:600;'>
                        {status}
                    </span>
                    &nbsp;&nbsp;
                    <span style='color:#777; font-size:13px;'>ขนาด: {size}</span>
                </div>
                <div style='margin-top:8px; font-size:12px; color:#555;'>
                    path: best.pt
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        skp_path = os.path.join(MODEL_DIR, "random_forest.pkl")
        exists2  = os.path.exists(skp_path)
        size2    = f"{os.path.getsize(skp_path)/1e6:.1f} MB" if exists2 else "—"
        status2  = "🟢 พร้อมใช้งาน" if exists2 else "🔴 ไม่พบไฟล์"
        color2   = "#2ecc71" if exists2 else "#e94560"

        st.markdown(f"""
            <div class="result-card">
                <div style='font-size:18px; font-weight:700;'>🌲 Random Forest (.skp)</div>
                <div style='margin-top:10px; font-size:14px; color:#aaa;'>
                    รับ features จาก YOLO แล้วทำนายน้ำหนักเป็น กก.
                </div>
                <div style='margin-top:14px;'>
                    <span style='background:{color2}22; color:{color2};
                                 padding:4px 14px; border-radius:20px;
                                 font-size:13px; font-weight:600;'>
                        {status2}
                    </span>
                    &nbsp;&nbsp;
                    <span style='color:#777; font-size:13px;'>ขนาด: {size2}</span>
                </div>
                <div style='margin-top:8px; font-size:12px; color:#555;'>
                    path: random_forest.skp
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ─── Pipeline ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Pipeline การทำงาน")

    steps = [
        ("1", "📂", "รับภาพ",
         "รองรับ JPG/PNG/BMP/WEBP/TIFF และ ZIP ที่มีภาพอยู่ภายใน"),
        ("2", "🎯", "YOLO Detection",
         "ตรวจจับตำแหน่งหมูในภาพ สร้าง Bounding Box พร้อม confidence score"),
        ("3", "📐", "Feature Extraction",
         "สกัด features จาก bbox: width, height, area, aspect ratio, confidence"),
        ("4", "🌲", "Random Forest Predict",
         "นำ features เข้า RandomForest เพื่อทำนายน้ำหนัก (กก.)"),
        ("5", "📊", "แสดงผล & Export",
         "แสดงภาพ before/after, น้ำหนัก, รายการทั้งหมด และดาวน์โหลด Excel"),
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
    st.markdown("### 🛠️ การติดตั้ง & รันแอป")

    st.markdown("**วางไฟล์โมเดลในโฟลเดอร์เดียวกับ `app.py`:**")
    st.code("""
pig_weight_app/
├── app.py
├── best.pt               ← โมเดล YOLOv8
├── random_forest.skp     ← โมเดล RandomForest
├── requirements.txt
└── pages_src/
    ├── __init__.py
    ├── page_analyze.py
    └── page_about.py
    """, language="text")

    st.markdown("**ติดตั้ง dependencies:**")
    st.code("""
pip install -r requirements.txt
    """, language="bash")

    st.markdown("**รันแอป:**")
    st.code("""
streamlit run app.py
    """, language="bash")

    # ─── Features ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✨ ฟีเจอร์ทั้งหมด")

    features = [
        ("📁", "อัปโหลดหลายรูปพร้อมกันหรือ ZIP"),
        ("🎯", "YOLO ตรวจจับ bounding box อัตโนมัติ"),
        ("🌲", "RandomForest ทำนายน้ำหนัก (กก.)"),
        ("🖼️", "แสดงภาพก่อน/หลังพร้อม layout"),
        ("📋", "รายการผลลัพธ์เรียงตามน้ำหนัก"),
        ("📥", "ดาวน์โหลดผลเป็น Excel (.xlsx)"),
        ("📊", "สรุป avg/max/min น้ำหนัก"),
        ("🔄", "Demo mode เมื่อยังไม่มีโมเดล"),
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