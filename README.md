# PrismNet: Compressed Image Classification (GB-03)

PrismNet is an optimized AI inference system designed for **Edge AI & Optimisation (AORUS Elite 16)**. It addresses the **GB-03** problem statement by compressing a baseline **ResNet-50** model using structured pruning and dynamic depth, achieving a **~65% reduction in model size** while improving inference speed on the **NVIDIA RTX 50-series**.

## 🎯 Problem Statement (GB-03)
> Design and build an optimised AI inference system by compressing a baseline model such as ResNet... achieving substantial model size reduction while maintaining accuracy and delivering faster inference.

## 🚀 Key Innovations

### 1. Structured L1-Norm Pruning
We've implemented channel-level pruning that removes the least important filters from the ResNet-50 backbone.
- **Outcome:** Direct reduction in model parameters and FLOPS.
- **Compression:** ~98MB (Baseline) → **~34MB (PrismNet)**.

### 2. Early-Exit Dynamic Depth
PrismNet introduces confidence-aware dynamic depth. Instead of running all 50 layers for every image, the system analyzes the scene at multiple "Exit Stages":
- **Stage 1 (Layer 1):** Ultra-fast exit for simple, high-confidence scenes.
- **Stage 2 (Layer 2):** Mid-depth exit for moderately complex scenes.
- **Stage 3 (Full):** Complete 50-layer inference for high-entropy images.

### 3. Blackwell GPU Acceleration
- **TensorFloat-32 (TF32):** Leveraging the new Blackwell math kernels for a 2x speedup over standard FP32.
- **Dynamic VRAM Scaling:** Real-time hardware detection and memory fraction allocation.

---

## 🔬 Optimization Deep-Dive

### Compression Pipeline
The compression is achieved through a two-stage process in `compression.py`:
1.  **Filter Pruning:** Convolutional layers are analyzed using the L1-norm of their weights. The bottom 30% of filters are permanently removed.
2.  **Model Distillation (Architecture):** The backbone is re-structured into an `EarlyExitResNet` which adds specialized classification heads after early blocks.

### Inference Performance
By combining pruning with Early-Exit, PrismNet achieves:
- **Baseline Latency:** ~100ms (on standard hardware).
- **PrismNet Latency:** **~20-40ms** (dynamic) on the RTX 50-series.
- **Size Reduction:** **65.3%** smaller footprint.

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

# 3. Launch the GB-03 Dashboard
python3 main.py --mode app
```

---

## 📊 Usage Guide
Launch the web interface at `http://localhost:8501`.
- **Optimization Mode:** Toggle between **Baseline** (Full ResNet50) and **Compressed** (PrismNet Pruned + Early Exit).
- **Early-Exit Threshold:** Adjust the AI's confidence requirement for stopping early.
- **Performance Analytics:** View real-time **Model Size**, **FPS**, and **Active Depth Path**.

**Project PrismNet** — *Optimized for the Edge. Built for AORUS.*
