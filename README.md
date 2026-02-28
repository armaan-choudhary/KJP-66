# PrismNet: SOTA On-Device Object Detection

PrismNet is a high-performance, on-device object detection system optimized for the **NVIDIA RTX 50-series (Blackwell)**. It features the latest **YOLO26** architecture combined with confidence-aware **Dynamic Resolution** to deliver extreme throughput without sacrificing accuracy.

## 🚀 Key Innovations

### 1. YOLO26 Integration (SOTA)
Leverages the January 2026 release of YOLO26, featuring **End-to-End NMS-Free inference**. This eliminates post-processing bottlenecks, ensuring deterministic latency and smoother performance on modern Tensor Cores.

### 2. Dynamic Resolution (Early Exit)
A confidence-aware depth system that adapts to scene complexity:
- **Stage 1 (Turbo):** Ultra-fast 320px inference for simple scenes (~5-10ms).
- **Stage 2 (Full):** Precise 640px inference for complex environments.
The system automatically "exits" at Stage 1 if high-confidence objects are detected, saving massive GPU cycles and battery.

### 3. Blackwell GPU Optimization
- **TensorFloat-32 (TF32):** Maximizes math throughput on RTX 50-series.
- **Dynamic VRAM Management:** Real-time allocation based on system-wide free memory.
- **BF16 Automatic Mixed Precision (AMP):** Utilizes Blackwell's optimized BFloat16 instructions.

---

## 🛠 Setup & Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU (RTX 50-series recommended)
- CUDA 12.8+

### Quick Start
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the PrismNet Dashboard
python3 main.py --mode app
```

---

## 📊 Usage Guide

### The Dashboard
Launch the web interface at `http://localhost:8501`.
- **Engine Class:** Toggle between `YOLO26n` (Turbo), `YOLO26m` (Performance), and `YOLO26x` (State-of-the-Art).
- **Inference Mode:** Switch between the standard **Baseline** and the optimized **PrismNet** pipeline.
- **Hardware Config:** Manually select your **Camera Device ID** if the default fails.

### Performance Benchmarking
To run a headless comparison across all optimization tiers:
```bash
python3 main.py --mode benchmark
```

---

## 📂 File Structure
- `st_app.py`: Modern, minimalist Streamlit dashboard.
- `early_exit.py`: `DynamicResolutionDetector` logic for multi-stage inference.
- `baseline.py`: Core YOLO loading and CUDA optimization layer.
- `main.py`: Unified CLI entry point for the entire system.

---

## 🛡 System Health
The real-time health monitor on the dashboard tracks:
- **Real-time FPS:** Actual throughput of the optimized pipeline.
- **Inference Latency:** Precise per-frame execution time.
- **VRAM Utilization:** Breakdown of PyTorch vs. System-wide memory usage.
- **Active Path:** Visual confirmation of the Dynamic Resolution stage.

**Project PrismNet** — *Designed for the AORUS Elite 16.*
