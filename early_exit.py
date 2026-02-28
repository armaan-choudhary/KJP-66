import torch
import cv2
import time
from ultralytics import YOLO

# PrismNet Project: Dynamic Resolution Detector (Object Detection Innovation)
print("--- PrismNet: Dynamic Detection System ---")

class DynamicResolutionDetector:
    """
    Simulates Early Exit in Object Detection by using dynamic input scaling.
    Shares the same model instance to conserve VRAM.
    """
    def __init__(self, model_instance, threshold=0.75):
        self.model = model_instance
        self.threshold = threshold
        
    def detect(self, frame):
        """
        Confidence-aware dynamic resolution inference for multi-object detection.
        """
        if frame is None or frame.size == 0:
            return None, 0, 0.0
            
        start = time.time()
        
        # Stage 1: Fast Inference at 320px
        # We use a lower confidence threshold internally to see if anything is there
        results1 = self.model.predict(frame, imgsz=320, conf=0.15, verbose=False)[0]
        
        # Multi-Object Analysis
        num_objects = len(results1.boxes)
        max_conf = 0.0
        if num_objects > 0:
            max_conf = results1.boxes.conf.max().item()
            
        # DECISION LOGIC: 
        # If we have a very clear high-confidence object, or multiple decent ones, exit.
        # Otherwise, upgrade to Stage 2 for clarity.
        if (num_objects >= 1 and max_conf >= self.threshold) or (num_objects >= 3 and max_conf > 0.5):
            latency = (time.time() - start) * 1000
            return results1, 1, latency
            
        # Stage 2: Full Inference at 640px for complex scenes
        results2 = self.model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        latency = (time.time() - start) * 1000
        return results2, 2, latency
