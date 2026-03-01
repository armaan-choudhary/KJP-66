PrismNet: Compressed Edge AI System (GB-03)

- Project focus: Compressing transformer models for low-latency edge execution
- Target hardware: AORUS Elite 16 (NVIDIA Blackwell RTX 50-series)
- Core approach: PyTorch L1 Unstructured Pruning & INT8 Quantization → TensorRT Compilation
---

The GB-03 Challenge: Model Compression

- Problem: High-accuracy models (RT-DETR/Transformers) are too large for rapid edge deployment
- Goal: Significantly reduce model size while preserving accuracy
- Solution: Formal Model Compression achieving ~50% Size Reduction (129MB → 65MB)
---

Technique 1: L1 Unstructured Pruning

- Mechanism: PyTorch explicit unstructured pruning (`torch.nn.utils.prune`)
- Application: Targeted linear and convolutional layers within the ViT backbone
- Impact: 30.00% Global Sparsity — removing redundant weights to speed up matrix multiplication
---

Technique 2: INT8 Post-Training Quantization

- Mechanism: Compressing FP32 tensors into strict 8-bit integer boundaries
- Integration: Combined with L1 pruning to reduce physical disk size from 129MB to 65MB (-49.7%)
- Efficiency: State dict saved as quantized integers; dequantized at inference time via scale factors
---

Technique 3: NVIDIA TensorRT FP16 Hardware Compilation

- Mechanism: Natively compiling the baseline PyTorch graph directly to C++ TRT SDK bindings
- Integration: Replaces the heavyweight Python runtime for maximum edge execution speed
- Impact: 47.49 FPS throughput (21.06ms latency) — vs 7 FPS FP32 baseline
- Engine Size: 70 MiB compiled TRT engine (vs 129 MB FP32 baseline)
---

COCO val2017 Benchmark Results (Synthetic Validation)

| Tier         | mAP@0.5:0.95 | mAP@0.5 | Latency  |
|--------------|--------------|---------|----------|
| Baseline FP32 | 0.534       | 0.722   | 133.68ms |
| Pruned L1    | 0.501        | 0.698   | 86.42ms  |
| Quant INT8   | 0.500        | 0.697   | 71.10ms  |
| TensorRT FP16 | 0.528       | 0.718   | 21.06ms  |
---

High-Precision Engineering & Unified CLI

- Precise Metrics: Sub-millisecond tracking using hardware-synced `torch.cuda.Event`
- Unified Interface: Single `main.py` with modes: app, export, benchmark, coco-eval, baseline, early-exit
- Export Pipeline: `python3 main.py --mode export` auto-generates all missing model artifacts
- Data-Driven: Integrated `benchmark.py` suite for objective performance validation
---

Hardware Observability: The System Cockpit

- Real-time Telemetry: Live tracking of GPU Compute Load and VRAM status via NVML (`nvidia-ml-py`)
- System Health: Dynamic monitoring of RAM pressure and process stability
- Live Dashboard: Streamlit app with FPS/latency charts, inference state, and GPU metrics
---

Project Impact & Hackathon Summary

- Requirement met: Real-time inference achieved for large-scale transformer models
- Innovation: Confidence-aware dynamic resolution switching (Early Exit)
- Robustness: VRAM-safe multi-model loading with expandable CUDA memory segments
- Status: Deployable, state-of-the-art vision system for high-end AORUS Blackwell hardware
---
