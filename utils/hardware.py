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
        # Prevent OpenCV from throwing C++ level FFMPEG warnings if device doesn't exist
        if not os.path.exists(f"/dev/video{i}"):
            continue
            
        # Suppress OpenCV warning logs for cleaner terminal output
        os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, i
            
    # Fallback to simulated camera if no hardware is found
    class DummyVideo:
        def __init__(self):
            import numpy as np
            self.np = np
            self.width = CAM_WIDTH
            self.height = CAM_HEIGHT
            self.frame_count = 0
            
        def isOpened(self): return True
        def release(self): pass
        def set(self, p, v): pass
        
        def read(self):
            frame = self.np.zeros((self.height, self.width, 3), dtype=self.np.uint8)
            # Create a moving shape to simulate live inference
            x = (self.frame_count * 8) % self.width
            cv2.circle(frame, (x, self.height // 2), 60, (67, 135, 255), -1)
            cv2.putText(frame, "NO CAMERA DETECTED - SIMULATION FEED", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, "GB-03 HARDWARE BENCHMARKING ACTIVE", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
            self.frame_count += 1
            return True, frame
            
    return DummyVideo(), "SIM"

def get_model_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024**2)
    return 0.0
