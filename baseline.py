import torch
import cv2
import time
import numpy as np
from ultralytics import YOLO

# PrismNet Project: Object Detection Baseline (YOLOv8)
print("--- PrismNet: Initializing Object Detection Baseline ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Targeting device: {device}")

def allocate_vram_dynamically():
    if not torch.cuda.is_available(): return
    try:
        # System-wide memory check
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_mib = free_bytes / (1024**2)
        total_mib = total_bytes / (1024**2)
        
        # Calculate a safe allocation fraction for this specific process
        # We want to use most of what's available but leave room for the display driver (RTX 50-series)
        target_mib = free_mib * 0.8  # Use 80% of current free VRAM
        fraction = min(0.95, target_mib / total_mib)
        
        # Set a hard limit to prevent OOMs and play nice with other GPU apps
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        
        print(f"--- PrismNet Dynamic GPU Allocation ---")
        print(f"Global Free: {free_mib:.0f}MB | Allocation Limit: {fraction*100:.1f}%")
        
        # Optimize internal allocator
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception as e:
        print(f"GPU Manager Warning: {e}")

allocate_vram_dynamically()

# OPTIMIZATION: Enable TensorFloat32 (TF32)
if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

def get_yolo_model(model_name='yolo11m.pt'):
    """
    Loads YOLO11m (Medium) for RTX 50-series.
    Medium model provides massive accuracy boost for the new Blackwell architecture.
    """
    print(f"Loading High-Performance {model_name}...")
    model = YOLO(model_name)
    model.to(device)
    
    # NOTE: torch.compile() removed due to compatibility issues with ultralytics predictor.
    # High-performance is maintained via TF32 and cuDNN benchmark settings.
            
    return model

def benchmark_detection(model, iterations=50):
    print(f"Benchmarking Object Detection on {device}...")
    dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(5):
        _ = model(dummy_input, verbose=False)
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input, verbose=False)
            
    avg_latency = (time.time() - start_time) / iterations * 1000
    print(f"Avg Detection Latency: {avg_latency:.2f} ms | FPS: {1000/avg_latency:.2f}")
    return avg_latency

if __name__ == "__main__":
    detector = get_yolo_model()
    benchmark_detection(detector)
