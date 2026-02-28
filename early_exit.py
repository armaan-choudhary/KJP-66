import torch
import time
from ultralytics import RTDETR

# PrismNet Project: Dynamic RT-DETR System (Innovation)
print("--- PrismNet: Dynamic RT-DETR System ---")

class DynamicRTDETR:
    """
    Implements a confidence-aware transformer scaler for RT-DETR.
    Optimizes the Vision Transformer (ViT) backbone based on input complexity.
    """
    def __init__(self, model_instance, threshold=0.75):
        self.model = model_instance
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def detect(self, frame):
        t0 = time.time()
        
        # 1. Fast Feature Extraction (320px)
        with torch.no_grad(), torch.autocast(device_type=self.device.type):
            res1 = self.model.predict(frame, imgsz=320, conf=0.25, verbose=False)[0]
            
        max_conf = 0.0
        if len(res1.boxes) > 0:
            max_conf = res1.boxes.conf.max().item()
            
        # Decision: If the Transformer Encoder is confident at lower resolution, exit.
        if max_conf >= self.threshold:
            lat = (time.time() - t0) * 1000
            return res1, 1, lat
            
        # 2. High-Precision Transformer Decoding (640px)
        with torch.no_grad(), torch.autocast(device_type=self.device.type):
            res2 = self.model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
            
        lat = (time.time() - t0) * 1000
        return res2, 2, lat
