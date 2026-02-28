import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import cv2
import time
import numpy as np
import os

# PrismNet Project: ResNet-50 Baseline (Object Classification - ImageNet)
print("--- PrismNet: Initializing ResNet-50 (ImageNet) ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def allocate_vram_dynamically():
    if not torch.cuda.is_available(): return
    try:
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except: pass

allocate_vram_dynamically()

def get_resnet50_imagenet(pretrained=True):
    """
    Loads ResNet-50 for 1000-class Object Recognition (ImageNet).
    Perfect baseline for GB-03 Edge AI compression.
    """
    if pretrained:
        print("Loading ImageNet-1K Weights...")
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(num_classes=1000)
            
    model = model.to(device)
    model.eval()
    return model

def benchmark_inference(model, iterations=50):
    print(f"Benchmarking Object Classification on {device}...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
        
    start_time = time.time()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
        for _ in range(iterations):
            _ = model(dummy_input)
            
    avg_latency = (time.time() - start_time) / iterations * 1000
    print(f"Avg Latency: {avg_latency:.2f} ms | FPS: {1000/avg_latency:.2f}")
    return avg_latency

if __name__ == "__main__":
    model = get_resnet50_imagenet()
    benchmark_inference(model)
