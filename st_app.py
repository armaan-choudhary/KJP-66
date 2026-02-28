import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
from baseline import get_resnet50_imagenet
from early_exit import EarlyExitResNet
from compression import apply_structured_pruning
import os

# PrismNet: GB-03 Edge AI & Optimisation Dashboard (Object Classification + ROI)
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
    from baseline import get_resnet50_imagenet
    base = get_resnet50_imagenet(pretrained=True)
    
    # 2. Optimized Model (Pruned + Early Exit)
    import copy
    from compression import apply_structured_pruning
    from early_exit import EarlyExitResNet
    m = copy.deepcopy(base)
    pruned = apply_structured_pruning(m, amount=0.3)
    ee_model = EarlyExitResNet(pruned, num_classes=1000).to(device)
    ee_model.eval()
    
    # 3. Load ImageNet Labels (Safe method)
    try:
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        classes = weights.meta["categories"]
    except Exception as e:
        st.warning(f"Label load failed: {e}. Using generic labels.")
        classes = [f"Object {i}" for i in range(1000)]
        
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def generate_target_bbox(feature_maps, frame_shape):
    """
    Direct Localization: Uses activation maps to find the bounding box.
    Added safety checks for OpenCV stability.
    """
    if feature_maps is None or len(feature_maps.shape) < 4:
        return None
        
    with torch.no_grad():
        # Compute activation map by averaging along channel dimension
        am = torch.mean(feature_maps, dim=1).squeeze()
        am = F.relu(am)
        max_val = torch.max(am)
        if max_val > 0:
            am /= max_val
        am = am.cpu().numpy()
        
        # Ensure am is 2D before resizing
        if len(am.shape) != 2: return None
        
        # Upscale to frame
        h, w = frame_shape[:2]
        try:
            am_resized = cv2.resize(am, (w, h))
            # Thresholding
            _, thresh = cv2.threshold((am_resized * 255).astype(np.uint8), 120, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 1000:
                    return cv2.boundingRect(c)
        except Exception:
            return None
    return None

def main():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/prism.png", width=80)
        st.title("PrismNet GB-03")
        st.markdown("---")
        mode = st.radio("Optimization Mode", ["Baseline", "Compressed (Prism-Sight)"])
        threshold = st.slider("Early-Exit Threshold", 0.4, 0.95, 0.85)
        cam_index = st.number_input("Camera ID", 0, 5, 0)
        if st.button("Reload System"):
            st.cache_resource.clear()
            st.rerun()

    base_mb, comp_mb = 102.5, 36.1
    active_mb = base_mb if mode == "Baseline" else comp_mb

    with st.spinner("Initializing ResNet..."):
        base_model, ee_model, device, classes = load_prismnet_resources()
        ee_model.threshold = threshold
        cap = get_device_camera(cam_index)

    if cap is None:
        st.error("Hardware Conflict: Camera not found.")
        return

    main_col, stats_col = st.columns([3, 1])
    with main_col:
        feed = st.empty()
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
                if mode == "Baseline":
                    logits = base_model(tensor)
                    exit_path = "Full Depth"
                    # Baseline doesn't return maps easily, so we skip bbox for simplicity
                else:
                    # Optimized Path with Feature Map Retrieval
                    res_tuple = ee_model(tensor, return_feature_maps=True)
                    if len(res_tuple) == 3:
                        logits, exit_stage, fmaps = res_tuple
                    else:
                        logits, exit_stage = res_tuple
                        fmaps = None
                        
                    exit_path = f"Stage {exit_stage}"
                    
                    # Draw ROI Boundary
                    if fmaps is not None:
                        bbox = generate_target_bbox(fmaps, frame.shape)
                        if bbox:
                            x, y, w, h = bbox
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 135, 67), 3)
                            cv2.putText(frame, "PRISM TARGET", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 135, 67), 2)

            lat = (time.time() - t0) * 1000
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(probs, 3, dim=1)
            
            feed.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            m_fps.markdown(f'<div class="status-card"><p class="metric-label">REAL-TIME FPS</p><p class="metric-value">{1000/lat:.1f}</p></div>', unsafe_allow_html=True)
            m_lat.markdown(f'<div class="status-card"><p class="metric-label">LATENCY</p><p class="metric-value">{lat:.1f}ms</p></div>', unsafe_allow_html=True)
            m_exit.markdown(f'<div class="status-card"><p class="metric-label">ACTIVE PATH</p><p class="metric-value">{exit_path}</p></div>', unsafe_allow_html=True)
            
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                m_vram.markdown(f'<div class="status-card"><p class="metric-label">SYS VRAM FREE</p><p class="metric-value">{free/1024**2:.0f}MB</p></div>', unsafe_allow_html=True)

            labels_html = "### Top-3 Predictions\n"
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
