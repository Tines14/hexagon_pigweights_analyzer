"""
Page 1 - วิเคราะห์น้ำหนักหมู
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
from datetime import datetime

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

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ─── Model loading (cached) ────────────────────────────────────────────────────
# หา root directory ของโปรเจกต์ (ที่เดียวกับ app.py)
def _build_search_paths(filename):
    """Create a list of all possible paths, including Streamlit Cloud."""
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


@st.cache_resource
def load_scaler():
    """Load the StandardScaler used during training"""
    scaler_path = _find_model("feature_scaler.pkl")
    if not JOBLIB_AVAILABLE or not scaler_path:
        return None
    return joblib.load(scaler_path)


@st.cache_resource
def load_selected_features():
    """Load the list of features used for training. (selected_features.pkl)"""
    sf_path = _find_model("selected_features.pkl")
    if not JOBLIB_AVAILABLE or not sf_path:
        return SELECTED_FEATURES  # fallback hardcoded list
    return joblib.load(sf_path)

# ─── Mask / Contour feature extraction ───────────────────────────────────────
# Feature order ตรงกับ selected_features ใน notebook:
# ['mask_area', 'Convex_Hull_Area', 'longest', 'perimeter', 'Hu_1', 'Hu_2', 'Hu_4']
SELECTED_FEATURES = ['mask_area', 'Convex_Hull_Area', 'longest', 'perimeter', 'Hu_1', 'Hu_2', 'Hu_4']


def _extract_mask_features(img_array, masks, idx, x1, y1, x2, y2):
    """
    สกัด 7 features จาก segmentation mask ของ YOLO (ตรงกับ notebook extract_2d_geometric_features):
    ['mask_area', 'Convex_Hull_Area', 'longest', 'perimeter', 'Hu_1', 'Hu_2', 'Hu_4']

    Hu moments = ค่าดิบจาก cv2.HuMoments (ไม่ใช้ log transform)
    longest = max(w, h) ของ minAreaRect (ตรง notebook)
    """
    import cv2
    h_img, w_img = img_array.shape[:2]

    # ─── สร้าง binary mask ────────────────────────────────────────────────────
    if masks is not None and idx < len(masks.data):
        mask_raw = masks.data[idx].cpu().numpy()
        mask_resized = cv2.resize(mask_raw, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        mask_float = (mask_resized > 0.5).astype(np.float32)
    else:
        mask_float = np.zeros((h_img, w_img), dtype=np.float32)
        mask_float[y1:y2, x1:x2] = 1.0

    # ─── หา contour (ใช้ CHAIN_APPROX_NONE ตาม notebook) ────────────────────
    mask_uint8 = (mask_float * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        w = float(x2 - x1); h = float(y2 - y1)
        return [w * h, w * h, max(w, h), 2 * (w + h), 0.0, 0.0, 0.0]

    contour = max(contours, key=cv2.contourArea)

    # ─── features ────────────────────────────────────────────────────────────
    mask_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    hull = cv2.convexHull(contour)
    convex_hull_area = float(cv2.contourArea(hull))

    # longest = max(w, h) ของ minAreaRect (ตรงกับ notebook)
    rect = cv2.minAreaRect(contour)
    w_r, h_r = rect[1]
    longest = float(max(w_r, h_r))

    # Hu Moments — ค่าดิบ ไม่ log (ตรงกับ notebook)
    M = cv2.moments(contour)
    hu = cv2.HuMoments(M).flatten()
    Hu_1 = float(hu[0])
    Hu_2 = float(hu[1])
    Hu_4 = float(hu[3])

    # ลำดับ: ['mask_area', 'Convex_Hull_Area', 'longest', 'perimeter', 'Hu_1', 'Hu_2', 'Hu_4']
    return [mask_area, convex_hull_area, longest, perimeter, Hu_1, Hu_2, Hu_4]


# ─── Mask cleaning (ตาม notebook clean_pig_mask) ─────────────────────────────
def clean_pig_mask(mask_float, use_blur=True):
    """Clean the mask using morphology (matching the notebook)."""
    import cv2
    mask_uint8 = (mask_float * 255).astype(np.uint8)
    if use_blur:
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (5, 5), 0)
        _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # เก็บเฉพาะ component ที่ใหญ่สุด
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_uint8 = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask_uint8


def mask_to_pil(mask_uint8):
    """Convert the binary mask to a PIL image (grayscale → RGB for display)."""
    from PIL import Image as PILImage
    return PILImage.fromarray(mask_uint8).convert("RGB")


# ─── Core analysis function ────────────────────────────────────────────────────
def analyze_pig_image(pil_image: Image.Image, filename: str,
                       yolo_model, rf_model, scaler=None, selected_features=None) -> dict:
    """
    วิเคราะห์รูปหมู 1 ตัว
    Returns dict: {filename, weight_kg, before_img, after_img, bbox_count}
    """
    import cv2
    img_array = np.array(pil_image.convert("RGB"))
    after_img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(after_img)

    bbox_count = 0
    features_list = []
    raw_mask_img   = None   # PIL: Raw Mask (Before Cleaning)
    clean_mask_img = None   # PIL: Clean Mask (After Cleaning)
    masked_img     = None   # PIL: Masked Image (Black Background)

    h_img, w_img = img_array.shape[:2]

    # ── YOLO inference ──────────────────────────────────────────────────────
    if yolo_model is not None:
        try:
            results = yolo_model(img_array, verbose=False)
            for r in results:
                masks = r.masks  # segmentation masks (None ถ้า YOLO detect-only)

                for idx, box in enumerate(r.boxes):
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

                    # ── สกัด raw mask ─────────────────────────────────────────
                    if masks is not None and idx < len(masks.data):
                        raw_np = masks.data[idx].cpu().numpy()
                        raw_resized = cv2.resize(raw_np, (w_img, h_img),
                                                  interpolation=cv2.INTER_LINEAR)
                        raw_bin = (raw_resized > 0.5).astype(np.float32)
                    else:
                        raw_bin = np.zeros((h_img, w_img), dtype=np.float32)
                        raw_bin[y1:y2, x1:x2] = 1.0

                    # raw mask image (before cleaning)
                    raw_uint8 = (raw_bin * 255).astype(np.uint8)
                    raw_mask_img = mask_to_pil(raw_uint8)

                    # clean mask (after cleaning)
                    clean_uint8 = clean_pig_mask(raw_bin, use_blur=True)
                    clean_mask_img = mask_to_pil(clean_uint8)

                    # masked image (black background)
                    clean_float = (clean_uint8 > 0).astype(np.float32)
                    mask_3ch = np.stack([clean_float] * 3, axis=-1)
                    masked_np = (img_array * mask_3ch).astype(np.uint8)
                    masked_img = Image.fromarray(masked_np)

                    # ── สกัด features จาก clean mask ──────────────────────────
                    feat = _extract_mask_features(img_array, masks, idx, x1, y1, x2, y2)
                    features_list.append(feat)

                    break  # ใช้เฉพาะ pig ตัวแรก (ตาม notebook)
        except Exception as e:
            st.warning(f"YOLO error: {e}")

    # ── RandomForest predict ─────────────────────────────────────────────────
    weight_kg = None
    if rf_model is not None and features_list:
        try:
            # features_dict เก็บค่าแต่ละ feature ตามชื่อ
            feat_names  = selected_features if selected_features else SELECTED_FEATURES
            feat_values = np.array(features_list).mean(axis=0)  # shape: (7,)
            # map ตามลำดับจาก SELECTED_FEATURES → feat_names
            feat_map = dict(zip(SELECTED_FEATURES, feat_values))
            ordered  = np.array([feat_map.get(f, 0.0) for f in feat_names]).reshape(1, -1)
            # scale ก่อน predict
            if scaler is not None:
                ordered = scaler.transform(ordered)
            weight_kg = float(rf_model.predict(ordered)[0])
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
        "raw_mask_img": raw_mask_img,
        "clean_mask_img": clean_mask_img,
        "masked_img": masked_img,
        "bbox_count": bbox_count,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
    }

# ─── Excel export ──────────────────────────────────────────────────────────────
def build_excel(results: list[dict]) -> bytes:
    if not EXCEL_AVAILABLE:
        return b""
    import pandas as pd
    rows = []
    for i, r in enumerate(results, 1):
        rows.append({
            "Pig ID": i,
            "File name": r["filename"],
            "Estimated weight (kg)": r["weight_kg"],
            "Number of detected bounding boxes": r["bbox_count"],
        })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Analysis results")
        ws = writer.sheets["Analysis results"]
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

def get_pig_stage(weight_kg):
    if weight_kg < 20:
        return "Pre-Piglet (< 20 kg)", "#888888"
    elif weight_kg < 35:
        return "Piglet stage (20–35 kg)", "#f39c12"
    elif weight_kg < 60:
        return "Growing stage (35–60 kg)", "#27ae60"
    else:
        return "Market stage (60 kg+)", "#e94560"

# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════
def render():
    # Header
    st.markdown("""
        <div class="page-header">
            <h1>📷 วิเคราะห์น้ำหนักสุกร</h1>
            <p>Upload group photos — supports single photos, multiple photos, or .zip files.</p>
        </div>
    """, unsafe_allow_html=True)

    # โหลดโมเดล
    yolo_model        = load_yolo()
    rf_model          = load_rf()
    scaler            = load_scaler()
    selected_features = load_selected_features()

    # สถานะโมเดล
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if yolo_model:
            st.success("✅ best.pt file loaded successfully.")
        else:
            st.warning("⚠️ best.pt not found — Use Demo mode.")
    with col_m2:
        if rf_model:
            st.success("✅ random_forest.pkl loaded successfully.")
        else:
            st.warning("⚠️ random_forest.pkl not found — Use Demo mode.")

    # ─── Debug info (ช่วย troubleshoot path บน Streamlit Cloud) ────────────────
    with st.expander("Debug: click to view"):
        import glob
        st.code(f"""
**Path information**
cwd          : {os.getcwd()}
__file__     : {os.path.abspath(__file__)}
best.pt found: {_find_model('best.pt') or 'NOT FOUND'}
rf.pkl found : {_find_model('random_forest.pkl') or 'NOT FOUND'}
YOLO_AVAILABLE : {YOLO_AVAILABLE}
JOBLIB_AVAILABLE : {JOBLIB_AVAILABLE}
yolo_model loaded: {yolo_model is not None}
rf_model loaded  : {rf_model is not None}
scaler loaded    : {scaler is not None}
selected_features: {selected_features}

files in cwd:
{chr(10).join(sorted(os.listdir(os.getcwd())))}

/mount/src exists: {os.path.isdir('/mount/src')}
{'/mount/src contents: ' + str(os.listdir('/mount/src')) if os.path.isdir('/mount/src') else ''}
""")

    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Upload zone ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "📂 Choose an image file or a ZIP file",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "zip"],
        accept_multiple_files=True,
        key=st.session_state.upload_key,
        help="Supports JPG, PNG, BMP, WEBP, TIFF, and .zip files containing images."
    )

    if uploaded:
        col_clear = st.columns([1, 1, 1])
        with col_clear[1]:
            if st.button("Clear Images", type="secondary", use_container_width=True):
                st.session_state.upload_key += 1
                st.rerun()

    if not uploaded:
        st.markdown("""
            <div style='text-align:center; color:#555; padding:40px 0;
                        border:2px dashed #2a2a4a; border-radius:16px; margin-top:24px;'>
                <div style='font-size:48px;'>🐷</div>
                <div style='font-size:14px; margin-top:12px;'>
                    There's no file yet — drag and drop the file or click the button above.
                </div>
            </div>
        """, unsafe_allow_html=True)
        return

    # ─── โหลดรูป ──────────────────────────────────────────────────────────────
    with st.spinner("⏳ Loading images..."):
        images = load_images_from_uploads(uploaded)

    if not images:
        st.error("❌ No images found in uploaded files — please check file formats.")
        return

    st.info(f"Found **{len(images)}** images — analyzing...")

    # ─── Analyze ──────────────────────────────────────────────────────────────
    results = []
    progress = st.progress(0, text="Processing images...")

    for i, (fname, img) in enumerate(images):
        result = analyze_pig_image(img, fname, yolo_model, rf_model, scaler, selected_features)
        results.append(result)
        progress.progress((i + 1) / len(images),
                          text=f"Processing {i+1}/{len(images)}: {fname}")
        time.sleep(0.05)

    progress.empty()
    st.success(f"Analysis completed for **{len(results)}** images")
    st.markdown("<hr style='border-color:#2a2a4a;'>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # แสดงผล
    # ═════════════════════════════════════════════════════════════════════════

    # ── Summary metrics ──────────────────────────────────────────────────────
    weights = [r["weight_kg"] for r in results]
    avg_w   = round(sum(weights) / len(weights))
    min_w   = round(min(weights))
    max_w   = round(max(weights))
    

    st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="val">{len(results)}</div>
                <div class="lbl">Total Images</div>
            </div>
            <div class="metric-card">
                <div class="val">{avg_w:.3f} kg</div>
                <div class="lbl">Average Weight</div>
            </div>
            <div class="metric-card">
                <div class="val">{min_w:.3f} kg</div>
                <div class="lbl">Minimum Weight</div>
            </div>
            <div class="metric-card">
                <div class="val">{max_w:.3f} kg</div>
                <div class="lbl">Maximum Weight</div>
        </div>
    """, unsafe_allow_html=True)


    # ── แสดงภาพตัวอย่าง (ภาพแรก) ─────────────────────────────────────────────
    st.markdown("**Example of analysis results.**")
    primary = results[0]

    # นับ stage ทุกรูป
    stage_counts = {"Pre-Piglet": 0, "Piglet": 0, "Growing": 0, "Market": 0}
    for r in results:
        w = r["weight_kg"]
        if w < 20:
            stage_counts["Pre-Piglet"] += 1
        elif w < 35:
            stage_counts["Piglet"] += 1
        elif w < 60:
            stage_counts["Growing"] += 1
        else:
            stage_counts["Market"] += 1

    stage_label, stage_color = get_pig_stage(primary['weight_kg'])

    # ── Row 1: Original + Raw Mask ─────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Original Image**")
        st.image(primary["before_img"], use_container_width=True)
    with col2:
        st.markdown("**Raw Mask (Before Cleaning)**")
        if primary["raw_mask_img"] is not None:
            st.image(primary["raw_mask_img"], use_container_width=True)
        else:
            st.image(primary["after_img"], use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Clean Mask + Masked Image ──────────────────────────────────
    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.markdown("**Clean Mask (After Cleaning)**")
        if primary["clean_mask_img"] is not None:
            st.image(primary["clean_mask_img"], use_container_width=True)
        else:
            st.info("No mask available (Demo mode)")
    with col4:
        st.markdown("**Masked Image (Black Background)**")
        if primary["masked_img"] is not None:
            st.image(primary["masked_img"], use_container_width=True)
        else:
            st.info("No mask available (Demo mode)")

    import streamlit.components.v1 as components

    stage_label, stage_color = get_pig_stage(primary['weight_kg'])

    st.markdown(f"""
        <div class="result-card" style="display:flex; align-items:center; gap:22px;">
            <div style="flex:1;">
                <div style='font-size:14px; color:#aaa;'>📁 {primary['filename']}</div>
                <div style='margin-top:6px; font-size:12px; color:#666;'>
                    🕐 Analyzed at: {primary['timestamp']}
                </div>
                <div style='margin-top:8px; font-size:14px;'>
                    Detected: <b>{primary['bbox_count']}</b> bounding box(es)
                </div>
                <div class="weight-badge">{primary['weight_kg']:.3f} kg</div>
                <div style='margin-top:10px; padding:8px 14px; border-radius:8px;
                            background:{stage_color}22; border:1px solid {stage_color};
                            color:{stage_color}; font-size:14px; font-weight:600; display:inline-block;'>
                    {stage_label}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    components.html(f"""
        <div style="text-align:center; font-family:sans-serif;">
            <div style="font-size:13px; color:#aaa; margin-bottom:8px;">Pig Stage Summary</div>
            <canvas id="stageChart" width="200" height="200"></canvas>
            <div style="margin-top:8px; font-size:14px; color:#aaa; line-height:2;">
                <span style="color:#888888;">●</span> Pre-Piglet: {stage_counts['Pre-Piglet']}&nbsp;&nbsp;
                <span style="color:#E8E80E;">●</span> Piglet: {stage_counts['Piglet']}<br>
                <span style="color:#ABE535;">●</span> Growing: {stage_counts['Growing']}&nbsp;&nbsp;
                <span style="color:#ED5C0E;">●</span> Market: {stage_counts['Market']}
            </div>
        </div>
        <script>
        var canvas = document.getElementById('stageChart');
        var ctx = canvas.getContext('2d');
        var data = [{stage_counts['Pre-Piglet']}, {stage_counts['Piglet']}, {stage_counts['Growing']}, {stage_counts['Market']}];
        var colors = ['#888888', '#E8E80E', '#ABE535', '#ED5C0E'];
        var total = data.reduce((a, b) => a + b, 0) || 1;
        var cx = 100, cy = 100, r = 85, inner = 50;
        var start = -Math.PI / 2;
        data.forEach(function(val, i) {{
            var slice = (val / total) * 2 * Math.PI;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.arc(cx, cy, r, start, start + slice);
            ctx.closePath();
            ctx.fillStyle = colors[i];
            ctx.fill();
            start += slice;
        }});
        ctx.beginPath();
        ctx.arc(cx, cy, inner, 0, 2 * Math.PI);
        ctx.fillStyle = '#40404D';
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 18px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(total, cx, cy);
        </script>
    """, height=300)

    # ── รายการภาพทั้งหมด (กรณีมีมากกว่า 1 ภาพ) ──────────────────────────────
    if len(results) > 1:
        st.markdown("---")
        st.markdown("All analysis results")

        # เรียงตามน้ำหนักมาก → น้อย
        sorted_results = sorted(results, key=lambda x: x["weight_kg"],
                                 reverse=True)

        for i, r in enumerate(sorted_results, 1):
        
            s_label, s_color = get_pig_stage(r['weight_kg'])
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
                        &nbsp;
                        <span style='font-size:12px; color:{s_color};'>
                            {s_label}
                        </span>
                    </div>
                    <div class="pig-wt">{r['weight_kg']:.3f} kg</div>
                </div>
            """, unsafe_allow_html=True)

    # ─── ดาวน์โหลด Excel ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("⬇️ Download Results")

    if EXCEL_AVAILABLE:
        excel_bytes = build_excel(results)
        st.download_button(
            label="⬇️ Download the Excel file (.xlsx).",
            data=excel_bytes,
            file_name="pig_weight_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.error("❌ pandas/openpyxl not found — Install with: `pip install pandas openpyxl`")

    # ปุ่มดาวน์โหลดภาพ after ของภาพแรก
    st.download_button(
        label="⬇️ Download sample image (after analysis)",
        data=pil_to_bytes(primary["after_img"]),
        file_name=f"analyzed_{primary['filename']}",
        mime="image/png",
        use_container_width=True,
    )