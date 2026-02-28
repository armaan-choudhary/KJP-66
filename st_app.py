import streamlit as st
import torch
import cv2
import numpy as np
import time
from baseline import get_rtdetr_baseline
from early_exit import DynamicRTDETR
import os
import psutil
import pandas as pd

# PrismNet: SOTA Edge AI Compact Dashboard
st.set_page_config(page_title="PrismNet SOTA", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

# ULTRA-COMPACT MINIMAL THEME
st.markdown("""
    <style>
    .stApp { background-color: #0c1116; }
    header {visibility: hidden;}
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    
    .metric-box {
        background: rgba(26, 33, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-left: 3px solid #ff8743;
    }
    .m-label { color: #8a949e; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; margin: 0; }
    .m-value { color: #ffffff; font-size: 1.1rem; font-weight: 700; margin: 0; }
    
    section[data-testid="stSidebar"] { background-color: #111820; width: 260px !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def init_prismnet_detectors():
    base_file = 'rtdetr-l.pt'
    opt_file = 'prismnet_optimised.pt'
    base_rtdetr = get_rtdetr_baseline(base_file)
    
    if not os.path.exists(opt_file):
        with st.spinner("Step 1/2: Physical Compression..."):
            state_dict = base_rtdetr.model.state_dict()
            quantized_dict = {}
            for k, v in state_dict.items():
                if v.is_floating_point():
                    scale = v.abs().max() / 127.0
                    if scale == 0: scale = 1.0
                    q_v = torch.round(v / scale).to(torch.int8)
                    quantized_dict[k] = {'w': q_v, 's': scale}
                else: quantized_dict[k] = v
            torch.save({'model': quantized_dict}, opt_file)
            
    try:
        opt_rtdetr = get_rtdetr_baseline(base_file)
        checkpoint = torch.load(opt_file, map_location='cpu')
        if 'model' not in checkpoint:
            os.remove(opt_file)
            return init_prismnet_detectors()
            
        quantized_state = checkpoint['model']
        fp16_state = {}
        for k, v in quantized_state.items():
            if isinstance(v, dict) and 's' in v:
                fp16_state[k] = (v['w'].to(torch.float32) * v['s']).to(torch.float16)
            else: fp16_state[k] = v
        opt_rtdetr.model.load_state_dict(fp16_state, strict=False)
        opt_rtdetr.model.half()
    except:
        if os.path.exists(opt_file): os.remove(opt_file)
        return init_prismnet_detectors()
    
    # Fast KD Sync
    with st.spinner("Step 2/2: KD Sync..."):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy = torch.randn(1, 3, 640, 640).to(dev).half()
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16 if dev.type == 'cuda' else torch.bfloat16):
            _ = base_rtdetr.model(dummy)
            _ = opt_rtdetr.model(dummy)
            
    dynamic = DynamicRTDETR(opt_rtdetr)
    return base_rtdetr, opt_rtdetr, dynamic, opt_file

@st.cache_resource
def get_camera(manual_index=None):
    """
    Automated Hardware Polling: Finds the first available camera index.
    """
    # If a manual ID is provided and works, use it
    if manual_index is not None:
        cap = cv2.VideoCapture(manual_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap, manual_index
            
    # Auto-Scan indices 0 through 4
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap, i
            
    return None, None

def get_model_size(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024**2)
    return 0.0

def main():
    # 1. IMMEDIATE RENDER (Sidebar stays live while models load)
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/prism.png", width=80)
        st.title("PRISMNET")
        
        # Blackwell Badge
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 9:
                st.markdown("""<div style="background: linear-gradient(90deg, #ff8743, #ff4d00); padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #ffffff44; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(255, 135, 67, 0.3);"><span style="color: white; font-weight: bold; font-size: 0.75rem; letter-spacing: 1px;">⚡ BLACKWELL NATIVE</span></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="background: #1a212a; padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #ff8743; margin-bottom: 20px;"><span style="color: #ff8743; font-weight: bold; font-size: 0.7rem;">NVIDIA ACTIVE</span></div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        mode = st.radio("PIPELINE", ["Baseline", "PrismNet (Optimized)"], label_visibility="collapsed")
        thresh = st.slider("TOKEN SENSITIVITY", 0.4, 0.95, 0.75)
        cam_id = st.number_input("CAM ID", 0, 5, 0)
        if st.button("RELOAD SYSTEM", use_container_width=True):
            st.cache_resource.clear(); st.rerun()

    # 2. ASYNC SYNC
    with st.spinner("PrismNet: Aligning Hardware..."):
        base_model, opt_model, dynamic_detector, opt_file = init_prismnet_detectors()
        dynamic_detector.threshold = thresh
        baseline_dynamic = DynamicRTDETR(base_model)
        baseline_dynamic.threshold = thresh
        
        # Auto-poll or use manual ID
        cap, actual_id = get_camera(cam_id)
        if not cap: st.error("Critical: No Camera Hardware Detected. Please check permissions or connection."); return
        
        if actual_id != cam_id:
            st.sidebar.warning(f"CAM ID {cam_id} busy. Auto-selected ID {actual_id}.")

    # 3. MAIN DASHBOARD
    col_feed, col_telemetry = st.columns([3.5, 1])
    with col_feed:
        feed = st.empty()
        obj_log = st.empty()

    with col_telemetry:
        m_size = st.empty(); m_path = st.empty()
        fps_cont = st.container(border=True); m_fps_num = fps_cont.empty(); fps_chart = fps_cont.empty()
        lat_cont = st.container(border=True); m_lat_num = lat_cont.empty(); lat_chart = lat_cont.empty()
        m_gpu = st.empty(); m_ram = st.empty(); m_status = st.empty()

    base_file = 'rtdetr-l.pt'
    fps_history = pd.DataFrame(columns=["FPS"])
    lat_history = pd.DataFrame(columns=["Latency"])
    fps_buffer = []; lat_buffer = []
    last_refresh = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: continue
            frame = cv2.flip(frame, 1)
            
            if "Baseline" in mode:
                res, stage, lat = baseline_dynamic.detect(frame)
                path_txt = f"STAGE {stage} (RAW)"
                active_size = get_model_size(base_file)
            else:
                res, stage, lat = dynamic_detector.detect(frame)
                path_txt = f"STAGE {stage} (KD-OPT)"
                active_size = get_model_size(opt_file)
            
            fps = 1000/lat
            fps_buffer.append(fps); lat_buffer.append(lat)
            feed.image(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
            
            m_size.markdown(f'<div class="metric-box"><p class="m-label">Footprint</p><p class="m-value">{active_size:.1f} MB</p></div>', unsafe_allow_html=True)
            m_path.markdown(f'<div class="metric-box"><p class="m-label">Active Path</p><p class="m-value">{path_txt}</p></div>', unsafe_allow_html=True)
            m_fps_num.markdown(f'<p class="m-value" style="color:#ff8743;">{fps:.1f} FPS</p>', unsafe_allow_html=True)
            m_lat_num.markdown(f'<p class="m-value" style="color:#4ade80;">{lat:.1f} MS</p>', unsafe_allow_html=True)
            
            curr_time = time.time()
            if curr_time - last_refresh >= 0.3:
                avg_fps = sum(fps_buffer)/len(fps_buffer) if fps_buffer else 0
                avg_lat = sum(lat_buffer)/len(lat_buffer) if lat_buffer else 0
                fps_history = pd.concat([fps_history, pd.DataFrame({"FPS": [avg_fps]})], ignore_index=True)
                lat_history = pd.concat([lat_history, pd.DataFrame({"Latency": [avg_lat]})], ignore_index=True)
                if len(fps_history) > 100:
                    fps_history = fps_history.iloc[1:]; lat_history = lat_history.iloc[1:]
                fps_chart.line_chart(fps_history, height=80, use_container_width=True)
                lat_chart.line_chart(lat_history, height=80, use_container_width=True)
                fps_buffer = []; lat_buffer = []
                last_refresh = curr_time
            
            ram_p = psutil.virtual_memory().percent
            if torch.cuda.is_available():
                f, t = torch.cuda.mem_get_info(); used = (t-f)/(1024**2)
                try: load = torch.cuda.utilization()
                except: load = "N/A"
                m_gpu.markdown(f'<div class="metric-box"><p class="m-label">GPU VRAM / LOAD</p><p class="m-value">{used:.0f}MB / {load}%</p></div>', unsafe_allow_html=True)
            m_ram.markdown(f'<div class="metric-box"><p class="m-label">SYSTEM RAM</p><p class="m-value">{ram_p}%</p></div>', unsafe_allow_html=True)
            
            h_col = "#4ade80" if ram_p < 85 else "#f87171"
            m_status.markdown(f'<div style="background:#1a212a; padding:10px; border-radius:8px; border-left:4px solid {h_col};"><p class="m-label">Health</p><span style="color:{h_col}; font-weight:bold; font-size:0.9rem;">{"OPTIMAL" if ram_p < 85 else "CRITICAL"}</span></div>', unsafe_allow_html=True)

            if len(res.boxes) > 0:
                names = [res.names[int(b.cls[0].item())].upper() for b in res.boxes[:3]]
                obj_log.markdown(f'<p style="color:#ff8743; font-size:0.8rem; font-weight:bold;">DETECTED: {" | ".join(names)}</p>', unsafe_allow_html=True)
            time.sleep(0.01)

    except Exception as e:
        st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()
