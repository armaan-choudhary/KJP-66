import torch
import time

class DynamicRTDETR:
    """
    Resolution-Aware Dynamic Inference Resolver.
    """
    def __init__(self, model_instance, threshold=0.75, is_optimized=True):
        self.model = model_instance
        self.threshold = threshold
        self.is_optimized = is_optimized
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def detect(self, frame):
        if torch.cuda.is_available():
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
        
        t0 = time.time()
        h, w = frame.shape[:2]
        aspect = w / h
        
        # Calculate Resolutions
        s1_h = max(320, (h // 2 // 32) * 32)
        s1_w = int((s1_h * aspect // 32) * 32)
        s2_h = min(640, (h // 32) * 32)
        s2_w = int((s2_h * aspect // 32) * 32)
        
        # Stage 1: Turbo Scan
        if self.is_optimized:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
                res1 = self.model.predict(frame, imgsz=(s1_h, s1_w), conf=0.25, verbose=False)[0]
        else:
            self.model.model.float()
            with torch.no_grad():
                res1 = self.model.predict(frame, imgsz=(s1_h, s1_w), conf=0.25, verbose=False)[0]
            
        max_conf = 0.0
        if len(res1.boxes) > 0:
            max_conf = res1.boxes.conf.max().item()
            
        if max_conf >= self.threshold:
            if torch.cuda.is_available():
                e.record(); torch.cuda.synchronize()
                latency = s.elapsed_time(e)
            else: latency = (time.time() - t0) * 1000
            return res1, 1, latency, f"{s1_w}x{s1_h}"
            
        # Stage 2: Precision Pass
        if self.is_optimized:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
                res2 = self.model.predict(frame, imgsz=(s2_h, s2_w), conf=0.25, verbose=False)[0]
        else:
            self.model.model.float()
            with torch.no_grad():
                res2 = self.model.predict(frame, imgsz=(s2_h, s2_w), conf=0.25, verbose=False)[0]
            
        if torch.cuda.is_available():
            e.record(); torch.cuda.synchronize()
            latency = s.elapsed_time(e)
        else: latency = (time.time() - t0) * 1000
            
        return res2, 2, latency, f"{s2_w}x{s2_h}"
