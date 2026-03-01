import streamlit as st

def apply_prism_theme():
    """
    Applies the ultra-compact Glassmorphism theme.
    """
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
        
        section[data-testid="stSidebar"] { background-color: #111820; width: 300px !important; }
        .sidebar-desc { color: #8a949e; font-size: 0.8rem; margin-bottom: 15px; line-height: 1.4; }
        </style>
    """, unsafe_allow_html=True)

def render_badge(mode="TensorRT (Accelerated)"):
    import torch
    
    if "Baseline" in mode:
        st.markdown("""<div style="background: rgba(138, 148, 158, 0.1); padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #8a949e; margin-bottom: 10px;"><span style="color: #8a949e; font-weight: bold; font-size: 0.8rem;">📦 FP32 BASELINE (UNOPTIMIZED)</span></div>""", unsafe_allow_html=True)
    elif "PrismNet" in mode:
        st.markdown("""<div style="background: rgba(74, 222, 128, 0.1); padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #4ade80; margin-bottom: 10px;"><span style="color: #4ade80; font-weight: bold; font-size: 0.8rem;">📦 L1 PRUNED & INT8 QUANTIZED (GB-03)</span></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background: rgba(59, 130, 246, 0.1); padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #3b82f6; margin-bottom: 10px;"><span style="color: #3b82f6; font-weight: bold; font-size: 0.8rem;">🚀 TENSORRT C++ ENGINE ACTIVE</span></div>""", unsafe_allow_html=True)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 9 or "RTX 50" in gpu_name:
            st.markdown("""<div style="background: linear-gradient(90deg, #ff8743, #ff4d00); padding: 12px; border-radius: 10px; text-align: center; border: 1px solid #ffffff44; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(255, 135, 67, 0.3);"><span style="color: white; font-weight: bold; font-size: 0.85rem; letter-spacing: 1px;">⚡ BLACKWELL NATIVE ACTIVE</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background: #1a212a; padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #ff8743; margin-bottom: 25px;"><span style="color: #ff8743; font-weight: bold; font-size: 0.8rem;">NVIDIA ACCELERATION ACTIVE</span></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background: #1a212a; padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #8a949e; margin-bottom: 25px;"><span style="color: #8a949e; font-weight: bold; font-size: 0.8rem;">CPU-ONLY MODE</span></div>""", unsafe_allow_html=True)
