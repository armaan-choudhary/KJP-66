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
    # Deduplicate indices to check
    indices_to_check = []
    if manual_index not in indices_to_check:
        indices_to_check.append(manual_index)
    for i in [0, 1, 2]:
        if i not in indices_to_check:
            indices_to_check.append(i)
            
    for i in indices_to_check:
        # Probe hardware path
        video_path = f"/dev/video{i}"
        if not os.path.exists(video_path):
            continue
            
        # Ensure it's a character device (cameras are char devices)
        if not os.stat(video_path).st_mode & 0o20000:
            continue
            
        # Silence OpenCV's V4L2 backend trace logs at the cv2 layer
        cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        
        if cap.isOpened():
            # Try a few times to get a frame, some cameras are slow to boot
            for _ in range(5):
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap, i
                import time
                time.sleep(0.1)
                
            cap.release()
                
    return None, None

def get_model_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024**2)
    return 0.0
