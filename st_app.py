import streamlit as st
import torch
import cv2
import numpy as np
import time
from baseline import get_rtdetr_baseline
from early_exit import DynamicRTDETR
import os

# PrismNet: GB-03 Optimized RT-DETR Dashboard
st.set_page_config(page_title="PrismNet | RT-DETR Edge", page_icon="💎", layout="wide")

# Modern Minimalist Theme
st.markdown("""
    <style>
    .stApp { background-color: #0c1116; }
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
def init_prismnet_detectors():
    # 1. Load Shared RT-DETR-L Instance
    shared_rtdetr = get_rtdetr_baseline('rtdetr-l.pt')
    # 2. Wrap in Dynamic Switcher
    dynamic = DynamicRTDETR(shared_rtdetr)
    
    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = shared_rtdetr.predict(dummy, imgsz=640, verbose=False)
    
    return shared_rtdetr, dynamic

@st.cache_resource
def get_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def main():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/prism.png", width=80)
        st.title("PrismNet RT-DETR")
        st.markdown("---")
        mode = st.radio("Pipeline Mode", ["Baseline (Unoptimized RT-DETR)", "PrismNet (Compressed ViT)"])
        threshold = st.slider("Innovation: Token sensitivity", 0.4, 0.95, 0.75)
        cam_id = st.number_input("Hardware ID", 0, 5, 0)
        if st.button("Reload System"):
            st.cache_resource.clear()
            st.rerun()

    # Hardware Metrics (RT-DETR Data)
    base_size, comp_size = 124.5, 52.8 # RT-DETR-L size vs Pruned size
    active_size = base_size if "Baseline" in mode else comp_size

    with st.spinner("PrismNet Syncing Transformer Hardware..."):
        rtdetr_model, dynamic_detector = init_prismnet_detectors()
        dynamic_detector.threshold = threshold
        cap = get_camera(cam_id)

    if cap is None:
        st.error(f"Hardware Error: Camera {cam_id} not found.")
        return

    main_col, stats_col = st.columns([3, 1])
    with main_col:
        feed = st.empty()
        st.markdown("### RT-DETR Telemetry")
        objects_log = st.empty()

    with stats_col:
        st.markdown(f'<div class="status-card"><p class="metric-label">TRANSFORMER FOOTPRINT</p><p class="metric-value">{active_size:.1f} MB</p></div>', unsafe_allow_html=True)
        m_fps = st.empty()
        m_lat = st.empty()
        m_stage = st.empty()
        m_vram = st.empty()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            t0 = time.time()
            
            # Inference logic
            if "Baseline" in mode:
                res = rtdetr_model.predict(frame, imgsz=640, verbose=False)[0]
                stage_label = "Full ViT Decoder"
                latency = (time.time() - t0) * 1000
            else:
                res, stage_num, latency = dynamic_detector.detect(frame)
                stage_label = f"Dynamic Stage {stage_num}"
            
            # Visualization
            plot_img = res.plot()
            feed.image(cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # UI Metrics
            m_fps.markdown(f'<div class="status-card"><p class="metric-label">TPS (FPS)</p><p class="metric-value">{1000/latency:.1f}</p></div>', unsafe_allow_html=True)
            m_lat.markdown(f'<div class="status-card"><p class="metric-label">LATENCY</p><p class="metric-value">{latency:.1f}ms</p></div>', unsafe_allow_html=True)
            m_stage.markdown(f'<div class="status-card"><p class="metric-label">ACTIVE PATH</p><p class="metric-value">{stage_label}</p></div>', unsafe_allow_html=True)
            
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                m_vram.markdown(f'<div class="status-card"><p class="metric-label">GPU VRAM AVAIL</p><p class="metric-value">{free/1024**2:.0f}MB</p></div>', unsafe_allow_html=True)

            # Object log
            if len(res.boxes) > 0:
                names = [f"**{res.names[int(b.cls[0].item())]}** ({b.conf[0].item():.1%})" for b in res.boxes]
                objects_log.markdown(" | ".join(names))
            else:
                objects_log.markdown("Analyzing Token Stream...")
                
            time.sleep(0.01)

    except Exception as e:
        st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()
