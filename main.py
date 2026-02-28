<<<<<<< HEAD
import os
import sys
import argparse

# PrismNet Project: Unified Entry Point
print("--- PrismNet: On-Device Scene Recognition System ---")
print("Target: AORUS Elite 16 (RTX 50-series)")

def main():
    parser = argparse.ArgumentParser(description="PrismNet: Optimized Scene Recognition Dashboard")
    parser.add_argument("--mode", type=str, choices=["baseline", "sensitivity", "early-exit", "app", "benchmark"], 
                        default="app", help="Run a specific part of the PrismNet pipeline")
    args = parser.parse_args()

    if args.mode == "baseline":
        import baseline
        model = baseline.get_resnet50_places365()
        baseline.benchmark_inference(model)
        
    elif args.mode == "sensitivity":
        import sensitivity
        import torch
        from baseline import get_resnet50_places365
        model = get_resnet50_places365(pretrained=False)
        calib = torch.randn(8, 3, 224, 224)
        report = sensitivity.measure_layer_sensitivity(model, calib)
        sensitivity.export_mixed_precision_model(model, report)
        
    elif args.mode == "early-exit":
        import early_exit
        import torch
        from baseline import get_resnet50_places365
        base = get_resnet50_places365(pretrained=False)
        ee_model = early_exit.EarlyExitResNet(base).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        early_exit.fine_tune_exits(ee_model)
        
    elif args.mode == "benchmark":
        import benchmark
        benchmark.run_full_benchmark()
        
    else: # app (default)
        import app
        app.launch_ui()

if __name__ == "__main__":
    main()
=======
print("Foooooooooo!!!!!!!!!!!!!")
>>>>>>> 950f3ab25695582ec6f465c152c56cbd7303119c
