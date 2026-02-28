import os
import sys
import argparse
import torch

# PrismNet Project: Unified Entry Point
print("--- PrismNet: On-Device RT-DETR Optimization ---")

def main():
    parser = argparse.ArgumentParser(description="PrismNet: Optimized Transformer Dashboard")
    parser.add_argument("--mode", type=str, choices=["baseline", "early-exit", "app", "benchmark"], 
                        default="app", help="Run a specific part of the PrismNet pipeline")
    args = parser.parse_args()

    if args.mode == "baseline":
        from baseline import get_rtdetr_baseline, benchmark_rtdetr
        model = get_rtdetr_baseline()
        benchmark_rtdetr(model)
        
    elif args.mode == "early-exit":
        # Demonstrates the dynamic resolution system
        from baseline import get_rtdetr_baseline
        from early_exit import DynamicRTDETR
        import cv2
        import numpy as np
        
        shared_rtdetr = get_rtdetr_baseline()
        dynamic = DynamicRTDETR(shared_rtdetr)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        res, stage, lat = dynamic.detect(dummy)
        print(f"Dynamic Test: Stage {stage} | Latency: {lat:.2f}ms")
        
    elif args.mode == "benchmark":
        import benchmark
        benchmark.run_full_benchmark()
        
    else: # app (default)
        print("Launching PrismNet Modern Minimal Dashboard...")
        os.system("streamlit run st_app.py")

if __name__ == "__main__":
    main()
