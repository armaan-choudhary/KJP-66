import torch
import cv2
import psutil
import os
from core.config import GPU_FRACTION, CAM_WIDTH, CAM_HEIGHT, DEFAULT_CAM_ID

def allocate_vram():
    """Optimizes GPU memory fraction."""
    if not torch.cuda.is_available(): return
    try:
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, 0)
    except: pass

def get_gpu_status():
    """Returns MiB used and Load percentage."""
    if not torch.cuda.is_available(): return 0, "N/A"
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        used = mem_info.used / (1024**2)
        load = utilization.gpu
        return used, load
    except: return 0, "N/A"

def get_system_cam(manual_index=DEFAULT_CAM_ID):
    """Auto-scans for camera hardware."""
    for i in [manual_index, 0, 1, 2]:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, i
    return None, None

def get_model_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024**2)
    return 0.0
