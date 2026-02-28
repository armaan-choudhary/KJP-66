import torch
import time
import json
import numpy as np
from core.engine import get_rtdetr_engine
from core.resolver import DynamicRTDETR
from utils.hardware import allocate_vram
import core.config as cfg

# PrismNet Project: Final Transformer Benchmark Suite
print("--- PrismNet: RT-DETR Performance Benchmark ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
allocate_vram()

def benchmark_rtdetr(model, iterations=30):
    print(f"Benchmarking RT-DETR on {device}...")
    dummy = np.random.randint(0, 255, (cfg.STAGE2_MAX_RES, cfg.STAGE2_MAX_RES, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(5):
        _ = model.predict(dummy, imgsz=cfg.STAGE2_MAX_RES, verbose=False)
        
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model.predict(dummy, imgsz=cfg.STAGE2_MAX_RES, verbose=False)
            
    lat = (time.time() - t0) / iterations * 1000
    print(f"Avg Latency: {lat:.2f}ms | FPS: {1000/lat:.1f}")
    return lat

def run_full_benchmark():
    results = {}
    
    # 1. Load Model
    shared_rtdetr = get_rtdetr_engine(cfg.MODEL_BASE)
    dynamic = DynamicRTDETR(shared_rtdetr)
    dummy = np.random.randint(0, 255, (cfg.STAGE2_MAX_RES, cfg.STAGE2_MAX_RES, 3), dtype=np.uint8)
    
    # 2. Benchmark Baseline (Fixed Max Res)
    print(f"\n[1/3] Benchmarking Baseline (Fixed {cfg.STAGE2_MAX_RES}px)...")
    t0 = time.time()
    for _ in range(30):
        _ = shared_rtdetr.predict(dummy, imgsz=cfg.STAGE2_MAX_RES, verbose=False)
    lat_base = (time.time() - t0) / 30 * 1000
    results[f"Baseline ({cfg.STAGE2_MAX_RES}px)"] = {"latency": round(lat_base, 2), "fps": round(1000/lat_base, 2)}
    
    # 3. Benchmark Turbo (Fixed Min Res)
    print(f"\n[2/3] Benchmarking Turbo (Fixed {cfg.STAGE1_MIN_RES}px)...")
    t0 = time.time()
    for _ in range(30):
        _ = shared_rtdetr.predict(dummy, imgsz=cfg.STAGE1_MIN_RES, verbose=False)
    lat_turbo = (time.time() - t0) / 30 * 1000
    results[f"Turbo ({cfg.STAGE1_MIN_RES}px)"] = {"latency": round(lat_turbo, 2), "fps": round(1000/lat_turbo, 2)}
    
    # 4. PrismNet Optimized (Dynamic)
    print("\n[3/3] Benchmarking PrismNet Dynamic Scaling...")
    # Simulate a mix of complexities
    t0 = time.time()
    for _ in range(30):
        _, _, _, _ = dynamic.detect(dummy)
    lat_prism = (time.time() - t0) / 30 * 1000
    results["PrismNet Optimized"] = {"latency": round(lat_prism, 2), "fps": round(1000/lat_prism, 2)}
    
    # Print Table
    print("\n" + "="*50)
    print(f"{'Model Variant':<25} | {'Latency (ms)':<12} | {'FPS':<8}")
    print("-" * 50)
    for name, stats in results.items():
        print(f"{name:<25} | {stats['latency']:<12.2f} | {stats['fps']:<8.2f}")
    print("="*50)
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    run_full_benchmark()
