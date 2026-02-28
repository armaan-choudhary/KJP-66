import streamlit as st
import torch
import cv2
import numpy as np
import time
from PIL import Image
from baseline import get_yolo_model
from early_exit import DynamicResolutionDetector
import os

# PrismNet: Modern Minimal Object Detection Dashboard
st.set_page_config(page_title="PrismNet Dashboard", page_icon="🎯", layout="wide")

# Custom CSS for Modern Minimal Aesthetic
st.markdown("""
    <style>
    .main { background-color: #0c1116; color: #ffffff; }
    .stMetric { background-color: #1a212a; padding: 15px; border-radius: 10px; border-left: 5px solid #ff8743; }
    .stRadio > label { color: #ff8743 !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_prismnet_detectors():
    # PRE-INIT: Clear memory cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_yolo = get_yolo_model('yolo11m.pt')
    base = shared_yolo
    dynamic = DynamicResolutionDetector(shared_yolo)
    
    # WARMUP
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = shared_yolo.predict(dummy, imgsz=640, verbose=False)
    _, _, _ = dynamic.detect(dummy)
    return base, dynamic, device

@st.cache_resource
def get_cap():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def main():
    st.title("PrismNet: On-Device Object Detection")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("PrismNet Settings")
    mode = st.sidebar.radio("Detection Mode", ["Baseline (Full 640px)", "Optimised (Dynamic Res)"])
    conf_thresh = st.sidebar.slider("Early-Exit Confidence", 0.5, 0.95, 0.75)
    
    # Load Models
    baseline_detector, dynamic_detector, device = load_prismnet_detectors()
    if mode == "Optimised (Dynamic Res)":
        dynamic_detector.threshold = conf_thresh

    # Layout: Camera Feed | Metrics
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Detection")
        img_placeholder = st.empty()
        
    with col2:
        st.subheader("Performance Analytics")
        metric_vram = st.empty()
        metric_objects = st.empty()
        metric_latency = st.empty()
        metric_stage = st.empty()
        metric_fps = st.empty()
        st.markdown("---")
        st.subheader("Detected Objects")
        objects_list = st.empty()

    # Camera Setup (Singleton)
    cap = get_cap()
    
    if cap is None:
        st.error("PrismNet Critical Error: Could not access camera hardware. Please check connection.")
        if st.button("Retry"): st.rerun()
        st.stop()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Start Inference Pipeline
            start_time = time.time()
            try:
                if "Baseline" in mode:
                    results = baseline_detector.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
                    stage = 2
                else:
                    results, stage, latency_raw = dynamic_detector.detect(frame)
                
                latency = (time.time() - start_time) * 1000
            except Exception as inf_err:
                st.warning(f"Inference Warning: {inf_err}")
                continue
            
            # Draw results using Ultralytics plot()
            plot_frame = results.plot()
            frame_rgb = cv2.cvtColor(plot_frame, cv2.COLOR_BGR2RGB)
            img_placeholder.image(frame_rgb, use_container_width=True)
            
            # Data Binding: Update UI
            num_objs = len(results.boxes)
            metric_objects.metric("Objects Found", num_objs)
            metric_latency.metric("Inference Latency", f"{latency:.2f} ms")
            metric_stage.metric("Inference Stage", f"Stage {stage}")
            metric_fps.metric("Optimized FPS", f"{1000/latency:.1f}")
            
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                vram_perc = (total - free) / total * 100
                metric_vram.metric("GPU VRAM Utilization", f"{vram_perc:.1f}%", f"{free/1024**2:.0f}MB Free")

            # Detected list
            obj_names = []
            if num_objs > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    name = results.names[cls_id]
                    obj_names.append(f"- **{name}** ({box.conf[0].item():.2%})")
            
            objects_list.markdown("\n".join(obj_names) if obj_names else "No objects detected.")
            
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"Stream interrupted: {e}")

if __name__ == "__main__":
    main()
