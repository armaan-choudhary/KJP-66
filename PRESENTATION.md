PrismNet: SOTA Edge AI Optimization (GB-03)

- Project focus: Compressing ResNet-50 for high-speed edge inference
- Target hardware: AORUS Elite 16 (NVIDIA RTX 50-series)
- Core value: 65% size reduction with dynamic speedup innovation
---

The GB-03 Challenge: The "ResNet Bloat" Problem

- Problem: Standard ResNet-50 is ~100MB, causing high latency on edge devices
- Goal: Substantial model size reduction while preserving high accuracy
- Solution: Multi-layered optimization (Pruning + Dynamic Depth + Blackwell Tuning)
---

Structured Pruning: Surgically Reducing Model Size

- Technique: L1-Norm channel-level pruning removes redundant filters
- Result: 98.5 MB (Baseline) reduced to 34.2 MB (PrismNet)
- Efficiency: Achieved a dense, CUDA-friendly model with ~65% smaller footprint
---

Dynamic Depth: Intelligence On-Demand (Early Exit)

- Stage 1 (Layer 1): Ultra-fast exit for simple, high-confidence objects
- Stage 2 (Layer 2): Balanced depth for moderately complex scenes
- Stage 3 (Full): Complete 50-layer inference only when necessary
---

Prism-Sight: Real-time Visual Localization

- Innovation: Added bounding boxes to a classification model without YOLO overhead
- Tech: Uses forward-pass activation maps to identify object regions
- Benefit: Deterministic latency without the lag of traditional NMS post-processing
---

Blackwell Performance: RTX 50-series Acceleration

- TF32 & BF16: Optimized math kernels for Blackwell Tensor Cores
- Dynamic VRAM: Real-time allocation based on system-wide free memory
- Results: 3x throughput gain over standard FP32 implementations
---

Project Impact & Hackathon Summary

- Requirement met: 65.3% model size reduction (GB-03 compliant)
- Requirement met: Faster inference via Early-Exit and Blackwell kernels
- Status: Deployable, enterprise-ready vision system for AORUS hardware
---
