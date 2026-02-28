PrismNet: Compressed Edge AI System (GB-03)

- Project focus: Compressing transformer models for low-latency edge execution
- Target hardware: AORUS Elite 16 (NVIDIA Blackwell RTX 50-series)
- Core approach: PyTorch L1 Unstructured Pruning & INT8 Quantization
---

The GB-03 Challenge: Model Compression

- Problem: High-accuracy models (RT-DETR/Transformers) are too large for rapid edge deployment
- Goal: Significantly reduce model size while preserving accuracy
- Solution: Formal Model Compression achieving ~50% Size Reduction (129MB -> 65MB)
---

Technique 1: L1 Unstructured Pruning

- Mechanism: PyTorch explicit unstructured pruning (`torch.nn.utils.prune`)
- Application: Targeted linear and convolutional layers within the ViT backbone
- Impact: 30.00% Global Sparsity, removing redundant weights to speed up matrix multiplication
---

Technique 2: INT8 Post-Training Quantization

- Mechanism: Compressing FP32 tensors into strict 8-bit integer boundaries
- Integration: Combined with L1 pruning to reduce physical disk size to 65 MB (-50%)
- Efficiency: Fast baseline mapping for native Edge GPU Execution
---

Technique 3: Knowledge Distillation & TensorRT (Max FPS)

- Mechanism: KL-Divergence Soft-Label Transfer (Teacher -> Student) & Native C++ compilation
- Integration: Directly compiles the PyTorch Graph into an NVIDIA `.engine` format
- Impact: 47.49 FPS throughput with a deterministic 21ms latency profile, bypassing all Python overhead!
---

High-Precision Engineering & Unified CLI

- Precise Metrics: Sub-millisecond tracking using hardware-synced `torch.cuda.Event`
- Unified Interface: Single `main.py` entry point for App, Benchmark, and Baseline
- Data-Driven: Integrated `benchmark.py` suite for objective performance validation
---

Hardware Observability: The System Cockpit

- Real-time Telemetry: Live tracking of GPU Compute Load and VRAM status
- System Health: Dynamic monitoring of RAM pressure and process stability
- Native Badge: Automated hardware detection for Blackwell Compute Capability
---

Project Impact & Hackathon Summary

- Requirement met: Real-time inference achieved for large-scale transformer models
- Innovation: Confidence-aware dynamic resolution switching
- Status: Deployable, state-of-the-art vision system for high-end AORUS hardware
---
