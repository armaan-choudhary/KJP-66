# PrismNet: Optimized RT-DETR for Edge AI (GB-03)

PrismNet is a state-of-the-art, transformer-based object detection system designed for **Edge AI & Optimisation (AORUS Elite 16)**. It addresses the **GB-03** problem statement by optimizing **RT-DETR (Real-Time Detection Transformer)**—the first end-to-end real-time transformer detector—using dynamic scaling and hardware-specific kernel tuning.

## 🚀 Key Innovations

### 1. NMS-Free Transformer Architecture
Unlike traditional YOLO models that require Non-Maximum Suppression (NMS) on the CPU, RT-DETR produces a fixed set of high-quality predictions natively.
- **Outcome:** Entire detection pipeline runs on the GPU with deterministic latency.

### 2. Dynamic Token Scaling (Early Exit)
Confidence-aware resolution switcher that optimizes the Vision Transformer (ViT) backbone:
- **Stage 1 (320px):** Ultra-fast transformer encoder pass (~10ms).
- **Stage 2 (640px):** Full-precision transformer decoding.

### 3. Blackwell GPU Optimization
- **TF32 & BF16 AMP:** Optimized math kernels for NVIDIA Blackwell.
- **Dynamic VRAM Scaling:** Real-time hardware monitoring and memory management.

### 4. High-Precision Engineering
- **Hardware-Synced Timing:** Uses `torch.cuda.Event` for sub-millisecond inference tracking.
- **Blackwell-Aware UI:** Dynamic badge system that detects and confirms Blackwell Native support.

---

## 🛡 Hardware Observability & Health
PrismNet provides a comprehensive "System Health" cockpit:
- **GPU Load Percentage:** Dynamic monitoring of Blackwell Tensor Core utilization.
- **Memory Footprint:** Precise breakdown of process-specific vs. system-wide VRAM.
- **Predictive Health Report:** Real-time analysis of resource pressure.

---

## 🛠 Setup & Installation

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch via Unified CLI
python3 main.py --mode app
```

---

## 📊 Unified CLI & Benchmarking
The project features a single entry point (`main.py`) for all operations:
- `--mode app`: Launches the modern Streamlit dashboard.
- `--mode benchmark`: Runs a full comparison suite across optimization tiers.
- `--mode baseline`: Tests the unoptimized transformer performance.
- `--mode early-exit`: Validates the dynamic resolution switching logic.

---

## 📂 File Structure
- `main.py`: Unified system entry point.
- `st_app.py`: Modern minimalist dashboard.
- `early_exit.py`: Dynamic Token Scaling implementation.
- `baseline.py`: Core RT-DETR and Blackwell optimization layer.
- `benchmark.py`: Data-driven performance analysis suite.

**Project PrismNet** — *The next generation of transformer-based edge vision.*
