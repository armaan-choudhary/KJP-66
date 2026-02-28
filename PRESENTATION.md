PrismNet: SOTA Transformer-Based Edge AI (GB-03)

- Project focus: Accelerating RT-DETR for real-time edge detection
- Target hardware: AORUS Elite 16 (NVIDIA Blackwell RTX 50-series)
- Core value: End-to-end NMS-free inference with dynamic token scaling
---

The GB-03 Challenge: Heavy-Duty Vision Transformers

- Problem: Transformers like RT-DETR are incredibly accurate but computationally intensive
- Goal: Maintain state-of-the-art accuracy while delivering real-time edge performance
- Solution: Dynamic scaling and native Blackwell hardware acceleration
---

NMS-Free Architecture: Eliminating the Bottleneck

- Innovation: RT-DETR is the first real-time end-to-end object detection transformer
- Technique: Natively produces high-quality predictions without CPU-bound NMS
- Impact: Achieving deterministic low-latency by keeping the entire pipeline on the GPU
---

Dynamic Token Scaling: Intelligence on Demand

- Stage 1 (320px): Fast transformer encoder pass for simple object presence
- Stage 2 (640px): High-precision decoding for complex, multi-object scenes
- Efficiency: Automatically scales computational depth based on image complexity
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
