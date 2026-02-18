"""
หน้าที่ 1 - วิเคราะห์น้ำหนักหมู
รองรับ: รูปเดียว / หลายรูป / ไฟล์ ZIP
โมเดล: best.pt (YOLOv8) + random_forest.pkl (RandomForest)
"""

import io
import os
import zipfile
import tempfile
import time
import random

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Try importing model libraries ────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import pandas as pd
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# ─── Model loading (cached) ────────────────────────────────────────────────────
# หา root directory ของโปรเจกต์ (ที่เดียวกับ app.py)
def _build_search_paths(filename):
    """สร้างรายการ path ที่เป็นไปได้ทั้งหมด รวมถึง Streamlit Cloud"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    paths = [
        filename,                                          # relative cwd
        os.path.join(cwd, filename),                      # absolute cwd
        os.path.join(script_dir, filename),               # same dir as script
        os.path.join(script_dir, "..", filename),         # parent of script
        os.path.join(script_dir, "..", "..", filename),   # grandparent
    ]

    # Streamlit Cloud เก็บ repo ไว้ที่ /mount/src/<github-user>/<repo-name>/
    for base in ["/mount/src", "/app", "/home/appuser"]:
        if os.path.isdir(base):
            # ระดับ base โดยตรง
            paths.append(os.path.join(base, filename))
            # ลง 1 ชั้น (username)
            try:
                for lvl1 in os.listdir(base):
                    p1 = os.path.join(base, lvl1)
                    paths.append(os.path.join(p1, filename))
                    # ลง 2 ชั้น (repo name)
                    if os.path.isdir(p1):
                        try:
                            for lvl2 in os.listdir(p1):
                                paths.append(os.path.join(p1, lvl2, filename))
                        except PermissionError:
                            pass
            except PermissionError:
                pass

    return paths


def _find_model(filename):
    for p in _build_search_paths(filename):
        try:
            if os.path.exists(p):
                return os.path.realpath(p)
        except Exception:
            continue
    return None


@st.cache_resource
def load_yolo():
    pt_path = _find_model("best.pt")
    if not YOLO_AVAILABLE or not pt_path:
        return None
    return YOLO(pt_path)  # ให้ exception ลอยขึ้นมาเองถ้าพัง (จะเห็นใน Streamlit log)


@st.cache_resource
def load_rf():
    rf_path = _find_model("random_forest.pkl")
    if not JOBLIB_AVAILABLE or not rf_path:
        return None
    return joblib.load(rf_path)

# ─── Core analysis function ────────────────────────────────────────────────────
def analyze_pig_image(pil_image: Image.Image, filename: str,
                       yolo_model, rf_model) -> dict:
    """
    วิเคราะห์รูปหมู 1 ตัว
    Returns dict: {filename, weight_kg, before_img, after_img, bbox_count}
    """
    img_array = np.array(pil_image.convert("RGB"))
    after_img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(after_img)

    bbox_count = 0
    features_list = []

    # ── YOLO inference ──────────────────────────────────────────────────────
    if yolo_model is not None:
        try:
            results = yolo_model(img_array, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    bbox_count += 1

                    # วาด bounding box
                    draw.rectangle([x1, y1, x2, y2],
                                   outline="#e94560", width=3)
                    draw.rectangle([x1, y1 - 24, x1 + 90, y1],
                                   fill="#e94560")
                    draw.text((x1 + 5, y1 - 20),
                              f"Pig {conf:.2f}",
                              fill="white")

                    # สกัด feature จาก bbox
                    w_px = x2 - x1
                    h_px = y2 - y1
                    area = w_px * h_px
                    ratio = w_px / (h_px + 1e-6)
                    features_list.append([w_px, h_px, area, ratio, conf])
        except Exception as e:
            st.warning(f"YOLO error: {e}")

    # ── RandomForest predict ─────────────────────────────────────────────────
    weight_kg = None
    if rf_model is not None and features_list:
        try:
            feat = np.array(features_list).mean(axis=0).reshape(1, -1)
            weight_kg = float(rf_model.predict(feat)[0])
        except Exception as e:
            st.warning(f"RF error: {e}")

    # ── Fallback: demo weight ────────────────────────────────────────────────
    if weight_kg is None:
        # จำลองน้ำหนักจาก pixel statistics (ใช้เมื่อยังไม่มีโมเดล)
        gray = np.mean(img_array)
        h, w = img_array.shape[:2]
        random.seed(int(gray * 100 + w + h))
        weight_kg = round(random.uniform(60, 140), 1)

        if bbox_count == 0:
            bbox_count = random.randint(1, 3)
            # วาด demo box
            margin = 40
            draw.rectangle([margin, margin,
                             pil_image.width - margin,
                             pil_image.height - margin],
                            outline="#e94560", width=3)
            draw.rectangle([margin, margin - 24,
                             margin + 90, margin],
                            fill="#e94560")
            draw.text((margin + 5, margin - 20),
                      f"Pig Demo", fill="white")

    return {
        "filename": filename,
        "weight_kg": weight_kg,
        "before_img": pil_image.convert("RGB"),
        "after_img": after_img,
        "bbox_count": bbox_count,
    }

# ─── Excel export ──────────────────────────────────────────────────────────────
def build_excel(results: list[dict]) -> bytes:
    if not EXCEL_AVAILABLE:
        return b""
    import pandas as pd
    rows = []
    for i, r in enumerate(results, 1):
        rows.append({
            "ลำดับ": i,
            "ชื่อไฟล์": r["filename"],
            "น้ำหนักโดยประมาณ (กก.)": r["weight_kg"],
            "จำนวน bbox ที่ตรวจพบ": r["bbox_count"],
        })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="ผลการวิเคราะห์")
        ws = writer.sheets["ผลการวิเคราะห์"]
        # ปรับความกว้างคอลัมน์
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = max_len + 4
    return buf.getvalue()

# ─── Image loader from uploaded files ─────────────────────────────────────────
def load_images_from_uploads(uploaded_files) -> list[tuple[str, Image.Image]]:
    """Returns list of (filename, PIL.Image)"""
    images = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    for uf in uploaded_files:
        ext = os.path.splitext(uf.name)[1].lower()

        if ext == ".zip":
            with zipfile.ZipFile(io.BytesIO(uf.read())) as zf:
                for zname in zf.namelist():
                    if os.path.splitext(zname)[1].lower() in image_exts:
                        with zf.open(zname) as img_file:
                            try:
                                img = Image.open(io.BytesIO(img_file.read()))
                                img.load()
                                images.append((os.path.basename(zname), img))
                            except Exception:
                                pass
        elif ext in image_exts:
            try:
                img = Image.open(io.BytesIO(uf.read()))
                img.load()
                images.append((uf.name, img))
            except Exception:
                pass

    return images

# ─── PIL image → bytes ────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════
def render():
    # Header
    st.markdown("""
        <div class="page-header">
            <h1>📷 วิเคราะห์น้ำหนักหมู</h1>
            <p>อัปโหลดภาพหมู — รองรับรูปเดี่ยว, หลายรูป, หรือไฟล์ .zip</p>
        </div>
    """, unsafe_allow_html=True)

    # โหลดโมเดล
    yolo_model = load_yolo()
    rf_model   = load_rf()

    # สถานะโมเดล
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if yolo_model:
            st.success("✅ โหลด best.pt สำเร็จ")
        else:
            st.warning("⚠️ ไม่พบ best.pt — ใช้โหมด Demo")
    with col_m2:
        if rf_model:
            st.success("✅ โหลด random_forest.pkl สำเร็จ")
        else:
            st.warning("⚠️ ไม่พบ random_forest.pkl — ใช้โหมด Demo")

    # ─── Debug info (ช่วย troubleshoot path บน Streamlit Cloud) ────────────────
    with st.expander("🔍 Debug: ข้อมูล Path (กดเพื่อดู)"):
        import glob
        st.code(f"""
cwd          : {os.getcwd()}
__file__     : {os.path.abspath(__file__)}
best.pt found: {_find_model('best.pt') or 'NOT FOUND'}
rf.pkl found : {_find_model('random_forest.pkl') or 'NOT FOUND'}
YOLO_AVAILABLE : {YOLO_AVAILABLE}
JOBLIB_AVAILABLE : {JOBLIB_AVAILABLE}
yolo_model loaded: {yolo_model is not None}
rf_model loaded  : {rf_model is not None}

files in cwd:
{chr(10).join(sorted(os.listdir(os.getcwd())))}

/mount/src exists: {os.path.isdir('/mount/src')}
{'/mount/src contents: ' + str(os.listdir('/mount/src')) if os.path.isdir('/mount/src') else ''}
""")


    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Upload zone ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "📂 เลือกไฟล์รูปภาพหรือไฟล์ ZIP",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "zip"],
        accept_multiple_files=True,
        help="รองรับ JPG, PNG, BMP, WEBP, TIFF และ .zip ที่มีรูปภาพอยู่ภายใน"
    )

    if not uploaded:
        st.markdown("""
            <div style='text-align:center; color:#555; padding:60px 0;
                        border:2px dashed #2a2a4a; border-radius:16px; margin-top:24px;'>
                <div style='font-size:48px;'>🐷</div>
                <div style='font-size:16px; margin-top:12px;'>
                    ยังไม่มีไฟล์ — ลากไฟล์มาวางหรือกดปุ่มด้านบน
                </div>
            </div>
        """, unsafe_allow_html=True)
        return

    # ─── โหลดรูป ──────────────────────────────────────────────────────────────
    with st.spinner("⏳ กำลังโหลดรูปภาพ..."):
        images = load_images_from_uploads(uploaded)

    if not images:
        st.error("❌ ไม่พบรูปภาพในไฟล์ที่อัปโหลด กรุณาตรวจสอบรูปแบบไฟล์")
        return

    st.info(f"📦 พบรูปทั้งหมด **{len(images)}** ภาพ — กำลังวิเคราะห์...")

    # ─── Analyze ──────────────────────────────────────────────────────────────
    results = []
    progress = st.progress(0, text="กำลังประมวลผล...")

    for i, (fname, img) in enumerate(images):
        result = analyze_pig_image(img, fname, yolo_model, rf_model)
        results.append(result)
        progress.progress((i + 1) / len(images),
                          text=f"ประมวลผล {i+1}/{len(images)}: {fname}")
        time.sleep(0.05)

    progress.empty()
    st.success(f"✅ วิเคราะห์เสร็จสิ้น {len(results)} ภาพ")
    st.markdown("<hr style='border-color:#2a2a4a;'>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # แสดงผล
    # ═════════════════════════════════════════════════════════════════════════

    # ── Summary metrics ──────────────────────────────────────────────────────
    weights = [r["weight_kg"] for r in results]
    avg_w   = round(sum(weights) / len(weights), 1)
    max_w   = max(weights)
    min_w   = min(weights)

    st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="val">{len(results)}</div>
                <div class="lbl">ภาพทั้งหมด</div>
            </div>
            <div class="metric-card">
                <div class="val">{avg_w} กก.</div>
                <div class="lbl">น้ำหนักเฉลี่ย</div>
            </div>
            <div class="metric-card">
                <div class="val">{max_w} กก.</div>
                <div class="lbl">น้ำหนักสูงสุด</div>
            </div>
            <div class="metric-card">
                <div class="val">{min_w} กก.</div>
                <div class="lbl">น้ำหนักต่ำสุด</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── แสดงภาพตัวอย่าง (ภาพแรก) ─────────────────────────────────────────────
    st.markdown("### 🖼️ ตัวอย่างผลการวิเคราะห์")
    primary = results[0]

    col_b, col_a = st.columns(2, gap="large")
    with col_b:
        st.markdown("**ก่อนวิเคราะห์**")
        st.image(primary["before_img"], use_container_width=True)
    with col_a:
        st.markdown("**หลังวิเคราะห์ (Layout)**")
        st.image(primary["after_img"], use_container_width=True)

    st.markdown(f"""
        <div class="result-card">
            <div style='font-size:15px; color:#aaa;'>📁 {primary['filename']}</div>
            <div style='margin-top:8px; font-size:14px;'>
                ตรวจพบ bbox: <b>{primary['bbox_count']}</b> ตำแหน่ง
            </div>
            <div class="weight-badge">🐷 {primary['weight_kg']} กก.</div>
        </div>
    """, unsafe_allow_html=True)

    # ── รายการภาพทั้งหมด (กรณีมีมากกว่า 1 ภาพ) ──────────────────────────────
    if len(results) > 1:
        st.markdown("---")
        st.markdown("### 📋 ผลการวิเคราะห์ทั้งหมด")

        # เรียงตามน้ำหนักมาก → น้อย
        sorted_results = sorted(results, key=lambda x: x["weight_kg"],
                                 reverse=True)

        for i, r in enumerate(sorted_results, 1):
            st.markdown(f"""
                <div class="pig-row">
                    <div>
                        <span style='color:#555; font-size:13px;'>#{i}</span>
                        &nbsp;
                        <span class="pig-name">{r['filename']}</span>
                        &nbsp;
                        <span style='color:#555; font-size:12px;'>
                            ({r['bbox_count']} bbox)
                        </span>
                    </div>
                    <div class="pig-wt">{r['weight_kg']} กก.</div>
                </div>
            """, unsafe_allow_html=True)

    # ─── ดาวน์โหลด Excel ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 ดาวน์โหลดผลลัพธ์")

    if EXCEL_AVAILABLE:
        excel_bytes = build_excel(results)
        st.download_button(
            label="⬇️  ดาวน์โหลดไฟล์ Excel (.xlsx)",
            data=excel_bytes,
            file_name="pig_weight_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.error("❌ ไม่พบ pandas / openpyxl — ติดตั้งด้วย: `pip install pandas openpyxl`")

    # ปุ่มดาวน์โหลดภาพ after ของภาพแรก
    st.download_button(
        label="🖼️  ดาวน์โหลดภาพตัวอย่าง (หลังวิเคราะห์)",
        data=pil_to_bytes(primary["after_img"]),
        file_name=f"analyzed_{primary['filename']}",
        mime="image/png",
        use_container_width=True,
    )