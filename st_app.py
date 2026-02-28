import streamlit as st
import torch
import cv2
import numpy as np
import time
from baseline import get_yolo_model
from early_exit import DynamicResolutionDetector
import os

# PrismNet: Ultra-Fast On-Device Detection Dashboard
st.set_page_config(page_title="PrismNet Turbo", page_icon="🚀", layout="wide")

# Minimalist PrismNet Theme
st.markdown("""
    <style>
    .stApp { background-color: #0c1116; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .status-card {
        padding: 20px;
        border-radius: 12px;
        background: #1a212a;
        border: 1px solid #2d3640;
        margin-bottom: 15px;
    }
    .metric-label { color: #8a949e; font-size: 0.9rem; }
    .metric-value { color: #ff8743; font-size: 1.8rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def init_prismnet_system(model_variant='yolo11n.pt'):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_yolo = get_yolo_model(model_variant)
    dynamic = DynamicResolutionDetector(shared_yolo)
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    _ = shared_yolo.predict(dummy, imgsz=320, verbose=False)
    return shared_yolo, dynamic, device

@st.cache_resource
def get_device_camera(index):
    try:
        cap = cv2.VideoCapture(index)
        if not cap or not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    except:
        return None

def main():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/prism.png", width=80)
        st.title("PrismNet")
        st.markdown("---")
        variant = st.selectbox("Engine Class", ["YOLO11n (Turbo)", "YOLO11m (High-End)"], index=0)
        mode = st.radio("Pipeline", ["Optimised (PrismNet)", "Baseline (Standard)"])
        threshold = st.slider("Early-Exit Sensitivity", 0.4, 0.95, 0.70)
        
        st.markdown("---")
        st.subheader("Hardware Config")
        cam_index = st.number_input("Camera Device ID", min_value=0, max_value=10, value=0)
        if st.button("Reload Hardware"):
            st.cache_resource.clear()
            st.rerun()
        
    model_file = 'yolo11n.pt' if "Turbo" in variant else 'yolo11m.pt'
    
    with st.spinner("PrismNet Initializing..."):
        baseline_detector, dynamic_detector, device = init_prismnet_system(model_file)
        dynamic_detector.threshold = threshold
        cap = get_device_camera(cam_index)

    if cap is None:
        st.error(f"Hardware Conflict: Camera {cam_index} not found. Try another Device ID.")
        return

    main_col, stats_col = st.columns([3, 1])
    with main_col:
        img_placeholder = st.empty()

    with stats_col:
        st.markdown('<div class="status-card"><p class="metric-label">SYSTEM STATUS</p><p style="color:#4ade80; font-weight:bold;">● ONLINE / ENCRYPTED</p></div>', unsafe_allow_html=True)
        m_fps = st.empty()
        m_lat = st.empty()
        m_exit = st.empty()
        m_objs = st.empty()
        m_vram = st.empty()

    try:
        while True:
            success, frame = cap.read()
            if not success: continue
            
            frame = cv2.flip(frame, 1)
            t0 = time.time()
            
            if mode == "Baseline (Standard)":
                res = baseline_detector.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
                stage = "Final"
            else:
                res, stage_num, _ = dynamic_detector.detect(frame)
                stage = f"Stage {stage_num}"
            
            latency = (time.time() - t0) * 1000
            img_placeholder.image(res.plot()[:,:,::-1], use_container_width=True)
            
            m_fps.markdown(f'<div class="status-card"><p class="metric-label">FPS</p><p class="metric-value">{1000/latency:.1f}</p></div>', unsafe_allow_html=True)
            m_lat.markdown(f'<div class="status-card"><p class="metric-label">LATENCY</p><p class="metric-value">{latency:.1f}ms</p></div>', unsafe_allow_html=True)
            m_exit.markdown(f'<div class="status-card"><p class="metric-label">ACTIVE PATH</p><p class="metric-value">{stage}</p></div>', unsafe_allow_html=True)
            m_objs.markdown(f'<div class="status-card"><p class="metric-label">OBJECTS</p><p class="metric-value">{len(res.boxes)}</p></div>', unsafe_allow_html=True)
            
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                vram_used = (total - free) / (1024**2)
                m_vram.markdown(f'<div class="status-card"><p class="metric-label">VRAM USED</p><p class="metric-value">{vram_used:.0f}MB</p></div>', unsafe_allow_html=True)

            time.sleep(0.001)

    except Exception as e:
        st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()
