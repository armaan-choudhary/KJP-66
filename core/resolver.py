import torch
import time
from core.config import DEFAULT_THRESHOLD, STAGE1_MIN_RES, STAGE2_MAX_RES, DETECTION_CONF, MODEL_STRIDE, PRECISION_BASELINE, PRECISION_OPTIMIZED

class DynamicRTDETR:
    """
    Resolution-Aware Dynamic Inference Resolver.
    """
    def __init__(self, model_instance, threshold=DEFAULT_THRESHOLD, is_optimized=True):
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
        s1_h = max(STAGE1_MIN_RES, (h // 2 // MODEL_STRIDE) * MODEL_STRIDE)
        s1_w = int((s1_h * aspect // MODEL_STRIDE) * MODEL_STRIDE)
        s2_h = min(STAGE2_MAX_RES, (h // MODEL_STRIDE) * MODEL_STRIDE)
        s2_w = int((s2_h * aspect // MODEL_STRIDE) * MODEL_STRIDE)
        
        # Stage 1: Turbo Scan
        if self.is_optimized:
            # OPTIMIZED: Uses INT8-Compressed Weights + Autocast FP16/BF16 Math
            torch.set_float32_matmul_precision(PRECISION_OPTIMIZED) # Allow TF32/Optimizations
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
                res1 = self.model.predict(frame, imgsz=(s1_h, s1_w), conf=DETECTION_CONF, verbose=False)[0]
        else:
            # BASELINE: Force Full Pure FP32 Precision (No Optimizations)
            torch.set_float32_matmul_precision(PRECISION_BASELINE)
            self.model.model.float()
            with torch.no_grad():
                res1 = self.model.predict(frame, imgsz=(s1_h, s1_w), conf=DETECTION_CONF, verbose=False)[0]
            
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
            torch.set_float32_matmul_precision(PRECISION_OPTIMIZED)
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
                res2 = self.model.predict(frame, imgsz=(s2_h, s2_w), conf=DETECTION_CONF, verbose=False)[0]
        else:
            torch.set_float32_matmul_precision(PRECISION_BASELINE)
            self.model.model.float()
            with torch.no_grad():
                res2 = self.model.predict(frame, imgsz=(s2_h, s2_w), conf=DETECTION_CONF, verbose=False)[0]
            
        if torch.cuda.is_available():
            e.record(); torch.cuda.synchronize()
            latency = s.elapsed_time(e)
        else: latency = (time.time() - t0) * 1000
            
        return res2, 2, latency, f"{s2_w}x{s2_h}"
