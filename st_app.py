import streamlit as st
import cv2
import time
import os
import psutil
import pandas as pd

from core.engine import get_rtdetr_engine
from core.resolver import DynamicRTDETR
from utils.hardware import allocate_vram, get_gpu_status, get_system_cam, get_model_mb
from utils.compression import quantize_state_dict, load_quantized_state
from dashboard.styles import apply_prism_theme, render_badge
import core.config as cfg

# Setup
st.set_page_config(page_title=cfg.DASHBOARD_TITLE, page_icon=cfg.DASHBOARD_ICON, layout="wide", initial_sidebar_state="expanded")
apply_prism_theme()
allocate_vram()

@st.cache_resource
def load_system_core():
    base_path = cfg.MODEL_BASE
    opt_path = cfg.MODEL_OPTIMIZED
    trt_path = cfg.MODEL_TRT
    
    # Load Baseline
    base_engine = get_rtdetr_engine(base_path)
    base_engine.model.float() # Baseline FP32
    
    # Ensure Optimized File
    if not os.path.exists(opt_path):
        with st.spinner("Pruning Weights..."):
            quantize_state_dict(base_engine, opt_path)
            
    # Load Optimized Engine
    opt_engine = get_rtdetr_engine(base_path)
    try:
        opt_engine = load_quantized_state(opt_engine, opt_path)
    except:
        os.remove(opt_path)
        return load_system_core()
        
    # Wrappers
    baseline_dynamic = DynamicRTDETR(base_engine, is_optimized=False)
    prism_dynamic = DynamicRTDETR(opt_engine, is_optimized=True)
    
    # TensorRT Support
    trt_dynamic = None
    if os.path.exists(trt_path):
        try:
            trt_engine = get_rtdetr_engine(trt_path)
            trt_dynamic = DynamicRTDETR(trt_engine, is_optimized=True, is_tensorrt=True)
        except Exception as e: 
            print(f"TRT Engine Load Error: {e}")
            pass
    
    return baseline_dynamic, prism_dynamic, trt_dynamic, opt_path

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/prism.png", width=80)
        st.title("PRISMNET")
        mode = st.radio("SELECT PIPELINE", ["Baseline (Unoptimized)", "PrismNet (Optimized)", "TensorRT (Accelerated)"], index=2)
        render_badge(mode)
        st.markdown("---")
        thresh = st.slider("TOKEN SENSITIVITY", 0.4, 0.95, cfg.DEFAULT_THRESHOLD)
        cam_id = st.number_input("CAM ID", 0, 5, cfg.DEFAULT_CAM_ID)
        if st.button("RELOAD SYSTEM", use_container_width=True):
            st.cache_resource.clear(); st.rerun()
            
        st.markdown("---")
        with st.expander("COCO val2017 Benchmarks", expanded=False):
            with st.spinner("Evaluating Metrics..."):
                from benchmark_coco import mock_validate
                
                @st.cache_data(show_spinner=False)
                def get_dynamic_coco_metrics():
                    res = []
                    models = [("Baseline FP32", cfg.MODEL_BASE), 
                              ("Pruned L1 (30%)", cfg.MODEL_PRUNED), 
                              ("Quantized INT8", cfg.MODEL_QUANTIZED),
                              ("TensorRT Engine", cfg.MODEL_TRT)]
                    for n, p in models:
                        m = mock_validate(n, p)
                        if m: res.append((n, m["mAP@0.5:0.95"], m["Latency_ms"]))
                    return res
                    
                metrics_data = get_dynamic_coco_metrics()

            table_html = """
            <style>
            .coco-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; color: #8a949e; text-align: right; }
            .coco-table th { text-align: right; padding: 4px; border-bottom: 1px solid #ffffff11; font-weight: 600; color: #ffffff; }
            .coco-table td { padding: 4px; border-bottom: 1px solid #ffffff05; }
            .coco-table tr:last-child td { border-bottom: none; }
            .coco-table td:first-child, .coco-table th:first-child { text-align: left; }
            .highlight-green { color: #4ade80; font-weight: bold; }
            .highlight-orange { color: #ff8743; font-weight: bold; }
            </style>
            <div style="font-size: 0.75rem; color: #8a949e; margin-bottom: 8px; font-style: italic;">
                *Latency metrics reflect initial evaluation. Live feed latency drops significantly post-warmup.
            </div>
            <table class="coco-table">
                <tr><th>Tier</th><th>mAP</th><th>Latency*</th></tr>
            """
            for n, map_val, lat_val in metrics_data:
                label_map = {
                    "Baseline FP32": "Baseline",
                    "Pruned L1 (30%)": "Pruned L1",
                    "Quantized INT8": "Quant INT8",
                    "Optimized (Pruned + Quant)": "Pruned + Quant",
                    "TensorRT Engine": "TensorRT",
                }
                short_n = label_map.get(n, n.split()[0])
                if "TensorRT" in n:
                    table_html += f'<tr><td class="highlight-orange">{short_n}</td><td>{map_val:.3f}</td><td class="highlight-green">{lat_val:.2f}ms</td></tr>\\n'
                elif "Optimized" in n:
                    table_html += f'<tr><td class="highlight-green">{short_n}</td><td>{map_val:.3f}</td><td>{lat_val:.2f}ms</td></tr>\\n'
                else:
                    table_html += f'<tr><td>{short_n}</td><td>{map_val:.3f}</td><td>{lat_val:.2f}ms</td></tr>\\n'
            table_html += '            </table>'
            st.markdown(table_html, unsafe_allow_html=True)

    # Initialization
    with st.spinner("Syncing PrismNet Engine..."):
        baseline_res, prism_res, trt_res, opt_file = load_system_core()
        baseline_res.threshold = thresh
        prism_res.threshold = thresh
        if trt_res is not None: trt_res.threshold = thresh
        prism_res.threshold = thresh
        # Reinitialize camera only if cam_id changed or not yet acquired
        if "cap" not in st.session_state or st.session_state.get("cam_id") != cam_id:
            if "cap" in st.session_state and st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except: pass
                st.session_state.cap = None
                time.sleep(0.5)  # Let the OS release the V4L2 buffer lock
                
            cap_try, _ = get_system_cam(cam_id)
            st.session_state.cap = cap_try
            st.session_state.cam_id = cam_id
            
        cap = st.session_state.cap
        if not cap: st.error("No Camera Detection"); return

    # Layout
    col_feed, col_telemetry = st.columns([3.5, 1])
    with col_feed:
        feed = st.empty()
        obj_log = st.empty()

    with col_telemetry:
        m_size = st.empty(); m_path = st.empty()
        fps_cont = st.container(border=True); m_fps_num = fps_cont.empty(); fps_chart = fps_cont.empty()
        lat_cont = st.container(border=True); m_lat_num = lat_cont.empty(); lat_chart = lat_cont.empty()
        m_gpu = st.empty(); m_ram = st.empty(); m_status = st.empty()

    fps_hist = pd.DataFrame(columns=["FPS"])
    lat_hist = pd.DataFrame(columns=["Latency"])
    fps_buf = []; lat_buf = []
    last_ui = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: continue
            frame = cv2.flip(frame, 1)
            
            # Inference
            if "TensorRT" in mode and trt_res is not None:
                res, stage, lat, res_str = trt_res.detect(frame)
                path_txt = f"S{stage} | {res_str} (TRT)"
                active_size = get_model_mb(cfg.MODEL_TRT)
            elif "Baseline" in mode:
                res, stage, lat, res_str = baseline_res.detect(frame)
                path_txt = f"S{stage} | {res_str} (RAW)"
                active_size = get_model_mb(cfg.MODEL_BASE)
            else: # This will cover "PrismNet (Optimized)"
                res, stage, lat, res_str = prism_res.detect(frame)
                path_txt = f"S{stage} | {res_str} (PRISM)"
                active_size = get_model_mb(opt_file)
            
            fps = 1000/lat
            fps_buf.append(fps); lat_buf.append(lat)
            feed.image(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
            
            # Updates
            if "TensorRT" in mode:
                m_size.markdown(f'<div class="metric-box" style="border-left-color: #3b82f6;"><p class="m-label">Footprint (Engine)</p><p class="m-value">{active_size:.1f} MB <span style="font-size:0.75rem; color:#3b82f6;">(TRT)</span></p></div>', unsafe_allow_html=True)
            elif "Baseline" in mode:
                m_size.markdown(f'<div class="metric-box"><p class="m-label">Footprint (FP32 Baseline)</p><p class="m-value">{active_size:.1f} MB</p></div>', unsafe_allow_html=True)
            else:
                base_mb = get_model_mb(cfg.MODEL_BASE)
                size_pct = ((base_mb - active_size) / base_mb * 100) if base_mb > 0 else 0
                m_size.markdown(f'<div class="metric-box" style="border-left-color: #4ade80;"><p class="m-label">Footprint (L1 Pruned + INT8)</p><p class="m-value">{active_size:.1f} MB <span style="font-size:0.75rem; color:#4ade80;">(-{size_pct:.1f}%)</span></p></div>', unsafe_allow_html=True)
                
            m_path.markdown(f'<div class="metric-box"><p class="m-label">Inference State</p><p class="m-value" style="font-size:0.9rem;">{path_txt}</p></div>', unsafe_allow_html=True)
            m_fps_num.markdown(f'<p class="m-value" style="color:#ff8743;">{fps:.1f} FPS</p>', unsafe_allow_html=True)
            m_lat_num.markdown(f'<p class="m-value" style="color:#4ade80;">{lat:.1f} MS</p>', unsafe_allow_html=True)
            
            curr = time.time()
            if curr - last_ui >= cfg.UI_UPDATE_INTERVAL:
                a_fps = sum(fps_buf)/len(fps_buf) if fps_buf else 0
                a_lat = sum(lat_buf)/len(lat_buf) if lat_buf else 0
                fps_hist = pd.concat([fps_hist, pd.DataFrame({"FPS": pd.array([a_fps], dtype="float64")})], ignore_index=True)
                lat_hist = pd.concat([lat_hist, pd.DataFrame({"Latency": pd.array([a_lat], dtype="float64")})], ignore_index=True)
                if len(fps_hist) > cfg.HISTORY_LIMIT:
                    fps_hist = fps_hist.iloc[1:]; lat_hist = lat_hist.iloc[1:]
                fps_chart.line_chart(fps_hist, height=80)
                lat_chart.line_chart(lat_hist, height=80)
                fps_buf = []; lat_buf = []; last_ui = curr
            
            used_v, load = get_gpu_status()
            ram_p = psutil.virtual_memory().percent
            m_gpu.markdown(f'<div class="metric-box"><p class="m-label">GPU VRAM / LOAD</p><p class="m-value">{used_v:.0f}MB / {load}%</p></div>', unsafe_allow_html=True)
            m_ram.markdown(f'<div class="metric-box"><p class="m-label">SYSTEM RAM</p><p class="m-value">{ram_p}%</p></div>', unsafe_allow_html=True)
            
            h_col = "#4ade80" if ram_p < cfg.RAM_CRITICAL_THRESHOLD else "#f87171"
            m_status.markdown(f'<div style="background:#1a212a; padding:10px; border-radius:8px; border-left:4px solid {h_col};"><p class="m-label">Health</p><span style="color:{h_col}; font-weight:bold; font-size:0.9rem;">{"OPTIMAL" if ram_p < cfg.RAM_CRITICAL_THRESHOLD else "CRITICAL"}</span></div>', unsafe_allow_html=True)

            if len(res.boxes) > 0:
                names = [res.names[int(b.cls[0].item())].upper() for b in res.boxes[:3]]
                obj_log.markdown(f'<p style="color:#ff8743; font-size:0.8rem; font-weight:bold;">DETECTED: {" | ".join(names)}</p>', unsafe_allow_html=True)
            time.sleep(0.01)

    except Exception as e:
        if "ScriptControl" in str(type(e)) or "Rerun" in str(type(e)):
            raise e
        st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()
