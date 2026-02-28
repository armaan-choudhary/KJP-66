# PrismNet: Optimized RT-DETR for Edge AI (GB-03)

PrismNet is a state-of-the-art, transformer-based object detection system designed for **Edge AI & Optimisation (AORUS Elite 16)**. It addresses the **GB-03** problem statement by optimizing **RT-DETR (Real-Time Detection Transformer)**—the first end-to-end real-time transformer detector—using dynamic scaling and hardware-specific kernel tuning.

## 🎯 Project Scope (GB-03)
This project focuses on the compression and acceleration of heavy-duty vision transformers. By leveraging the NMS-free architecture of RT-DETR, PrismNet delivers deterministic low-latency inference on the **NVIDIA RTX 50-series (Blackwell)**.

## 🚀 Key Innovations

### 1. NMS-Free Transformer Architecture
Unlike traditional YOLO models that require Non-Maximum Suppression (NMS) on the CPU, RT-DETR produces a fixed set of high-quality predictions natively.
- **Benefit:** Eliminates the post-processing bottleneck.
- **Outcome:** Entire detection pipeline runs on the GPU with deterministic latency.

### 2. Dynamic Token Scaling (Early Exit)
We've implemented a confidence-aware resolution switcher that optimizes the Vision Transformer (ViT) backbone:
- **Stage 1 (320px):** Ultra-fast transformer encoder pass for simple frames (~10ms).
- **Stage 2 (640px):** Full-precision transformer decoding for complex scenes.
The system intelligently scales tokens only when necessary, maximizing throughput on the edge.

### 3. Blackwell Hardware Acceleration
- **TF32 & BF16 AMP:** Optimized for the new math kernels in RTX 50-series.
- **Dynamic VRAM Scaling:** Real-time hardware monitoring and memory allocation.

---

## 🛠 Setup & Installation

### Quick Start
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Transformer Dashboard
python3 main.py --mode app
```

---

## 📊 Dashboard Features
Launch the interface at `http://localhost:8501`:
- **Optimization Mode:** Toggle between **Baseline** (Full RT-DETR) and **PrismNet** (Dynamic Scaling).
- **Token Sensitivity:** Adjust the confidence threshold for the early-resolution switch.
- **Transformer Telemetry:** Real-time monitoring of **Transformer Footprint**, **FPS**, and **Inference Stage**.

**Project PrismNet** — *The next generation of transformer-based edge vision.*
