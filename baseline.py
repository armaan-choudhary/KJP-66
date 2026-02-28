import torch
import cv2
import time
import numpy as np
from ultralytics import RTDETR

# PrismNet Project: RT-DETR Baseline (State-of-the-art Transformer)
print("--- PrismNet: Initializing RT-DETR Baseline ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def allocate_gpu_vram():
    if not torch.cuda.is_available(): return
    try:
        torch.cuda.set_per_process_memory_fraction(0.85, 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except: pass

allocate_gpu_vram()

def get_rtdetr_baseline(model_name='rtdetr-l.pt'):
    """
    Loads RT-DETR Large. 
    Transformers provide high accuracy but are heavy, perfect for GB-03 compression.
    """
    print(f"Syncing RT-DETR Hardware: {model_name}")
    # RT-DETR is end-to-end and NMS-free by design
    model = RTDETR(model_name)
    model.to(device)
    return model

def benchmark_rtdetr(model, iterations=30):
    print(f"Benchmarking RT-DETR on {device}...")
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(5):
        _ = model.predict(dummy, imgsz=640, verbose=False)
        
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model.predict(dummy, imgsz=640, verbose=False)
            
    lat = (time.time() - t0) / iterations * 1000
    print(f"Avg Latency: {lat:.2f}ms | FPS: {1000/lat:.1f}")
    return lat

if __name__ == "__main__":
    m = get_rtdetr_baseline()
    benchmark_rtdetr(m)
