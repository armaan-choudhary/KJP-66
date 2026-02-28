import torch
import time
import json
import numpy as np
from baseline import get_rtdetr_baseline
from early_exit import DynamicRTDETR

# PrismNet Project: Final Transformer Benchmark Suite
print("--- PrismNet: RT-DETR Performance Benchmark ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_full_benchmark():
    results = {}
    
    # 1. Load Model
    shared_rtdetr = get_rtdetr_baseline('rtdetr-l.pt')
    dynamic = DynamicRTDETR(shared_rtdetr)
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 2. Benchmark Baseline (Fixed 640px)
    print("\n[1/3] Benchmarking Baseline (Full Decoder)...")
    t0 = time.time()
    for _ in range(30):
        _ = shared_rtdetr.predict(dummy, imgsz=640, verbose=False)
    lat_base = (time.time() - t0) / 30 * 1000
    results["Baseline (640px)"] = {"latency": round(lat_base, 2), "fps": round(1000/lat_base, 2)}
    
    # 3. Benchmark Turbo (Fixed 320px)
    print("\n[2/3] Benchmarking Turbo (Encoder Only)...")
    t0 = time.time()
    for _ in range(30):
        _ = shared_rtdetr.predict(dummy, imgsz=320, verbose=False)
    lat_turbo = (time.time() - t0) / 30 * 1000
    results["Turbo (320px)"] = {"latency": round(lat_turbo, 2), "fps": round(1000/lat_turbo, 2)}
    
    # 4. PrismNet Optimized (Dynamic)
    print("\n[3/3] Benchmarking PrismNet Dynamic Scaling...")
    # Simulate a mix of complexities
    t0 = time.time()
    for _ in range(30):
        _, _, _ = dynamic.detect(dummy)
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
