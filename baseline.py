import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import time
import numpy as np
import os

# PrismNet Project: FP32 ResNet-50 Baseline + Benchmark
print("--- PrismNet: Initializing Baseline (FP32) ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Targeting device: {device}")

def get_resnet50_places365(pretrained=True):
    """
    Loads ResNet-50. Attempts to load Places365 weights if pretrained=True.
    Falls back to ImageNet or random initialization with 365 classes.
    """
    model = models.resnet50(num_classes=365)
    
    if pretrained:
        # Standard URL for Places365 ResNet50 weights (if accessible)
        # Note: In a real hackathon environment, you'd likely have these pre-downloaded.
        model_url = 'http://places2.csail.mit.edu/models_resnet50/resnet50_places365.pth.tar'
        try:
            print(f"Attempting to load Places365 weights from {model_url}...")
            checkpoint = torch.hub.load_state_dict_from_url(model_url, map_location=device)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            print("Successfully loaded Places365 weights.")
        except Exception as e:
            print(f"Could not load Places365 weights: {e}")
            print("Falling back to random initialization with 365 classes for demo purposes.")
    
    model = model.to(device)
    model.eval()
    
    # RTX 50-series optimization
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            print("Applying torch.compile() for RTX 50-series optimization...")
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}")
            
    return model

def benchmark_inference(model, input_size=(1, 3, 224, 224), iterations=100):
    print(f"Benchmarking inference on {device} over {iterations} iterations...")
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / iterations * 1000 # ms
    fps = 1000 / avg_latency
    print(f"Average Latency: {avg_latency:.2f} ms | FPS: {fps:.2f}")
    return avg_latency, fps

def run_camera_feed(model):
    print("Starting camera feed... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    
    # Simple labels placeholder (365 classes)
    # In practice, you'd load these from a categories_places365.txt file
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        
        img_tensor = torch.from_numpy(img).to(device).unsqueeze(0)
        
        start = time.time()
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
        latency = (time.time() - start) * 1000
        
        # Display
        label = f"PrismNet Baseline | Class: {pred.item()} | Conf: {conf.item():.2f} | {latency:.1f}ms"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('PrismNet Baseline', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    baseline_model = get_resnet50_places365()
    benchmark_inference(baseline_model)
    # run_camera_feed(baseline_model) # Commented out for headless benchmark script
