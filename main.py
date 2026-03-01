import os
import argparse

import core.config as cfg

# Entry Point
print("--- PRISMNET ---")

def main():
    parser = argparse.ArgumentParser(description="PrismNet: Optimized Transformer Dashboard")
    parser.add_argument("--mode", type=str, choices=["baseline", "early-exit", "app", "benchmark", "coco-eval", "export"], 
                        default="app", help="Run a specific part of the PrismNet pipeline")
    parser.add_argument("--coco-mode", type=str, choices=["baseline", "pruned", "quantized", "tensorrt", "all"], 
                        default="all", help="Model tier to evaluate for COCO metrics")
    parser.add_argument("--data-dir", type=str, default="datasets/coco", help="Path to COCO val2017 dataset")
    args = parser.parse_args()

    if args.mode == "baseline":
        from core.engine import get_rtdetr_engine
        from benchmark import benchmark_rtdetr
        model = get_rtdetr_engine(cfg.MODEL_BASE)
        benchmark_rtdetr(model)
        
    elif args.mode == "early-exit":
        # Test dynamic resolution system
        from core.engine import get_rtdetr_engine
        from core.resolver import DynamicRTDETR
        import numpy as np
        
        shared_rtdetr = get_rtdetr_engine(cfg.MODEL_BASE)
        dynamic = DynamicRTDETR(shared_rtdetr)
        dummy = np.zeros((cfg.STAGE2_MAX_RES, cfg.STAGE2_MAX_RES, 3), dtype=np.uint8)
        res, stage, lat, res_str = dynamic.detect(dummy)
        print(f"Dynamic Test: Stage {stage} | Resolution: {res_str} | Latency: {lat:.2f}ms")
        
    elif args.mode == "benchmark":
        import benchmark
        benchmark.run_full_benchmark()
        
    elif args.mode == "coco-eval":
        from benchmark_coco import run_coco_eval
        run_coco_eval(args.coco_mode, args.data_dir)
    
    elif args.mode == "export":
        from core.compressor import compress_rtdetr
        
        # Pruned + Quantized combined model
        if not os.path.exists(cfg.MODEL_OPTIMIZED):
            print(f"[EXPORT] Generating compressed model -> {cfg.MODEL_OPTIMIZED}")
            compress_rtdetr(source_path=cfg.MODEL_BASE, target_path=cfg.MODEL_OPTIMIZED,
                           sparsity=cfg.PRUNING_RATIO, run_tensorrt_export=False)
        else:
            print(f"[SKIP] {cfg.MODEL_OPTIMIZED} already exists.")
            
        # TensorRT Engine
        if not os.path.exists(cfg.MODEL_TRT):
            print(f"[EXPORT] Compiling TensorRT engine -> {cfg.MODEL_TRT}")
            compress_rtdetr(source_path=cfg.MODEL_BASE, target_path=cfg.MODEL_OPTIMIZED,
                           sparsity=cfg.PRUNING_RATIO, run_tensorrt_export=True)
        else:
            print(f"[SKIP] {cfg.MODEL_TRT} already exists.")
            
        print("\n[EXPORT] All model artifacts are up to date.")
        
    else: # app mode
        print("Launching dashboard...")
        os.system("streamlit run st_app.py")

if __name__ == "__main__":
    main()
