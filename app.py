import streamlit as st
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window

# Set up page configurations
st.set_page_config(page_title="SAR Anomaly Explorer", layout="wide")
st.title("🛰️ Sentinel-1 Deep SAR Anomaly Explorer")
st.subheader("Pixel-Geometry Structural Change Validation Engine")

# --- Hardcoded Project File Paths ---
M1_PATH = "data/S1A_IW_GRDH_1SDV_20260514T010316_20260514T010341_064506_081FE3_B4B1_COG.SAFE/measurement/s1a-iw-grd-vv-20260514t010316-20260514t010341-064506-081fe3-001-cog.tiff"
M2_PATH = "data/S1A_IW_GRDH_1SDV_20260619T010315_20260619T010340_065031_083245_3C65.SAFE/measurement/s1a-iw-grd-vv-20260619t010315-20260619t010340-065031-083245-001.tiff"
JSON_PATH = "anomalies.json"

@st.cache_data
def load_anomaly_log(path):
    with open(path, "r") as f:
        return json.load(f)

try:
    anomaly_records = load_anomaly_log(JSON_PATH)
    st.sidebar.success(f"Loaded {len(anomaly_records)} structural anomalies from log.")
except Exception as e:
    st.sidebar.error(f"Could not load log file: {e}")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.markdown("### Select Anomaly Instance")
anomaly_ids = [item["id"] for item in anomaly_records]
selected_id = st.sidebar.selectbox("Choose Target Anomaly ID:", anomaly_ids)

# Extract matching metadata footprint
target = next(item for item in anomaly_records if item["id"] == selected_id)
bbox = target["pixel_bounding_box"]
area_size = target["pixel_area_size"]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Anomaly Metrics:**")
st.sidebar.write(f"* Contiguous Pixels: `{area_size}`")
st.sidebar.write(f"* Column Range: `{bbox['col_start']}` to `{bbox['col_end']}`")
st.sidebar.write(f"* Row Range: `{bbox['row_start']}` to `{bbox['row_end']}`")

# --- Dynamic Patch Slicing Core ---
# Calculate bounding envelope sizes with padding to add geographic context
col_w = bbox["col_end"] - bbox["col_start"]
row_h = bbox["row_end"] - bbox["row_start"]
padding = 64  # Add a buffer around the anomaly for better visualization context

win = Window(
    max(0, bbox["col_start"] - padding),
    max(0, bbox["row_start"] - padding),
    col_w + (padding * 2),
    row_h + (padding * 2)
)

@st.cache_data(show_spinner="Extracting array slices from source tiffs...")
def fetch_target_windows(win_tuple):
    # Convert window tuples back to rasterio Window objects
    window_obj = Window(*win_tuple)
    with rasterio.open(M1_PATH) as src1, rasterio.open(M2_PATH) as src2:
        t1_slice = src1.read(1, window=window_obj).astype(np.float32)
        t2_slice = src2.read(1, window=window_obj).astype(np.float32)
    return t1_slice, t2_slice

# Streamlit caching requires standard python tuples instead of custom objects
slice_t1, slice_t2 = fetch_target_windows((win.col_off, win.row_off, win.width, win.height))

# Apply Decibel math transformations for enhanced visual profiling
db_t1 = 20 * np.log10(np.abs(slice_t1) + 1e-6)
db_t2 = 20 * np.log10(np.abs(slice_t2) + 1e-6)

# Compute localized delta difference matrix
delta_diff = db_t2 - db_t1

# --- Rendering Layout ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📅 May 2026 (Time 1 Baseline)")
    fig, ax = plt.subplots()
    ax.imshow(db_t1, cmap="gray", vmin=-25, vmax=0)
    ax.axis("off")
    st.pyplot(fig)

with col2:
    st.markdown("#### 📅 June 2026 (Time 2 Check)")
    fig, ax = plt.subplots()
    ax.imshow(db_t2, cmap="gray", vmin=-25, vmax=0)
    ax.axis("off")
    st.pyplot(fig)

with col3:
    st.markdown("#### 📊 Mathematical Delta Variance (T2 - T1)")
    fig, ax = plt.subplots()
    # Diverging colormap highlights changes (e.g., bright for new builds, dark for clearing)
    ax.imshow(delta_diff, cmap="bwr", vmin=-10, vmax=10)
    ax.axis("off")
    st.pyplot(fig)

st.markdown("---")
st.info("💡 **Interpretation Guide:** In the Delta Variance map, bright blue segments show an increase in radar backscatter (common in new urban construction or ship arrivals). Bright red areas show a drop in backscatter (common in clearing vegetation, ground grading, or flooding events).")
