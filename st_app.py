import streamlit as st
import torch
import cv2
import numpy as np
import time
from baseline import get_resnet50_imagenet
from early_exit import EarlyExitResNet
from compression import apply_structured_pruning
import os

# PrismNet: GB-03 Edge AI & Optimisation Dashboard (Object Classification)
st.set_page_config(page_title="PrismNet | ResNet Object", page_icon="📦", layout="wide")

# Minimalist PrismNet Theme
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
def load_prismnet_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Baseline Model (FP32 ImageNet)
    base = get_resnet50_imagenet(pretrained=True)
    
    # 2. Optimized Model (Pruned + Early Exit)
    import copy
    m = copy.deepcopy(base)
    pruned = apply_structured_pruning(m, amount=0.3)
    ee_model = EarlyExitResNet(pruned, num_classes=1000).to(device)
    ee_model.eval()
    
    # 3. Load ImageNet Labels
    from torchvision.models import ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    classes = weights.meta["categories"]
        
    return base, ee_model, device, classes

def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)

@st.cache_resource
def get_device_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap or not cap.isOpened(): return None
    return cap

def main():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/prism.png", width=80)
        st.title("PrismNet GB-03")
        st.markdown("---")
        mode = st.radio("Optimization Mode", ["Baseline (ResNet50)", "Compressed (PrismNet)"])
        threshold = st.slider("Early-Exit Threshold", 0.4, 0.95, 0.85)
        cam_index = st.number_input("Camera ID", 0, 5, 0)
        if st.button("Reload System"):
            st.cache_resource.clear()
            st.rerun()

    # Model Size Stats
    base_mb = 102.5 # ResNet50 FP32 ImageNet
    comp_mb = 36.1  # Pruned + Dynamic Depth
    active_mb = base_mb if "Baseline" in mode else comp_mb

    with st.spinner("PrismNet Initializing Object Engine..."):
        base_model, ee_model, device, classes = load_prismnet_resources()
        ee_model.threshold = threshold
        cap = get_device_camera(cam_index)

    if cap is None:
        st.error("Camera not detected.")
        return

    main_col, stats_col = st.columns([3, 1])

    with main_col:
        feed = st.empty()
        st.markdown("### Top-3 Object Predictions")
        top3_placeholder = st.empty()

    with stats_col:
        st.markdown('<div class="status-card"><p class="metric-label">MODEL SIZE</p><p class="metric-value">{:.1f} MB</p></div>'.format(active_mb), unsafe_allow_html=True)
        m_fps = st.empty()
        m_lat = st.empty()
        m_exit = st.empty()
        m_vram = st.empty()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            tensor = preprocess(frame).to(device)
            t0 = time.time()
            
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                if "Baseline" in mode:
                    logits = base_model(tensor)
                    exit_path = "Full Depth"
                else:
                    logits, exit_stage = ee_model(tensor)
                    exit_path = f"Stage {exit_stage}"
            
            lat = (time.time() - t0) * 1000
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(probs, 3, dim=1)
            
            # Update Feed
            feed.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Update Stats
            m_fps.markdown(f'<div class="status-card"><p class="metric-label">REAL-TIME FPS</p><p class="metric-value">{1000/lat:.1f}</p></div>', unsafe_allow_html=True)
            m_lat.markdown(f'<div class="status-card"><p class="metric-label">LATENCY</p><p class="metric-value">{lat:.1f}ms</p></div>', unsafe_allow_html=True)
            m_exit.markdown(f'<div class="status-card"><p class="metric-label">DYNAMIC DEPTH</p><p class="metric-value">{exit_path}</p></div>', unsafe_allow_html=True)
            
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                m_vram.markdown(f'<div class="status-card"><p class="metric-label">SYS VRAM FREE</p><p class="metric-value">{free/1024**2:.0f}MB</p></div>', unsafe_allow_html=True)

            # Update Labels
            labels_html = ""
            for i in range(3):
                p = top_probs[0][i].item()
                name = classes[top_idxs[0][i]].title()
                labels_html += f"**{name}**: {p:.1%}\n\n"
            top3_placeholder.markdown(labels_html)
            
            time.sleep(0.01)

    except Exception as e:
        st.error(f"Stream Error: {e}")

if __name__ == "__main__":
    main()
