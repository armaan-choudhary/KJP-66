import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import time
import numpy as np
import os

# PrismNet Project: ResNet-50 Baseline (Image Classification)
print("--- PrismNet: Initializing ResNet-50 Baseline ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def allocate_vram_dynamically():
    if not torch.cuda.is_available(): return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except: pass

allocate_vram_dynamically()

def get_resnet50_places365(pretrained=True):
    """
    Loads ResNet-50 for 365-class scene recognition.
    Essential for GB-03 Edge AI compression tasks.
    """
    model = models.resnet50(num_classes=365)
    
    if pretrained:
        model_url = 'https://huggingface.co/pytorch/vision/resolve/main/resnet50_places365.pth'
        try:
            print("Loading Places365 Weights...")
            checkpoint = torch.hub.load_state_dict_from_url(model_url, map_location=device, check_hash=False)
            if 'state_dict' in checkpoint:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            else:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Weights load failed: {e}. Using random init for demo.")
            
    model = model.to(device)
    model.eval()
    return model

def benchmark_inference(model, iterations=50):
    print(f"Benchmarking Classification on {device}...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
        
    start_time = time.time()
    with torch.no_grad(), torch.autocast(device_type=device.type):
        for _ in range(iterations):
            _ = model(dummy_input)
            
    avg_latency = (time.time() - start_time) / iterations * 1000
    print(f"Avg Latency: {avg_latency:.2f} ms | FPS: {1000/avg_latency:.2f}")
    return avg_latency

if __name__ == "__main__":
    model = get_resnet50_places365()
    benchmark_inference(model)
