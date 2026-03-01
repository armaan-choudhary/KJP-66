# PrismNet: Compressed RT-DETR for Edge AI (GB-03)

PrismNet is a custom-engineered **AI Inference System** that directly addresses the **GB-03** problem statement by radically compressing a high-accuracy baseline model (RT-DETR). The solution applies rigorous formal model compression techniques—achieving significant file size reduction and deterministic latency for the **AORUS Elite 16 (Blackwell RTX 50-series)** edge appliance.

## 🚀 Key Innovations (Model Compression)

### 1. PyTorch L1 Unstructured Pruning
We apply formal `torch.nn.utils.prune` L1 Unstructured Pruning logic specifically to the linear and convolutional layers within the Transformer architecture, achieving a **30.0% global sparsity** rate while preserving complex detection logic.

### 2. Post-Training INT8 Quantization
Following pruning, the entire state dictionary undergoes INT8 Quantization, compressing large floating-point arrays into strict integer bounds.
- **Outcome:** Model size drops from the FP32 Baseline (129.47 MB) to **65.12 MB** — a **~50% physical size reduction** compliant with GB-03's low-latency specification.

### 3. NVIDIA TensorRT Hardware Compilation
To shatter the FPS ceiling, the baseline Vision Transformer is natively compiled into a C++ TensorRT `.engine` file via `core/compressor.py`.
- **Outcome:** Bypassing the Python engine runtime entirely, the TRT pipeline hits an astonishing **47.49 FPS** (21.06ms latency) — a colossal leap over the 7 FPS FP32 baseline.

---

## � Model Optimization Results

| Artifact | File | Size | vs Baseline |
|---|---|---|---|
| Baseline FP32 | `rtdetr-x.pt` | 130 MB | — |
| Pruned + Quant INT8 | `prismnet_compressed.pt` | 66 MB | **-49.7%** |
| TensorRT FP16 Engine | `rtdetr-x.engine` | 71 MB | **-45.4%** |

### COCO val2017 Benchmark

| Tier | mAP@0.5:0.95 | mAP@0.5 | Latency (cold) |
|---|---|---|---|
| Baseline FP32 | 0.534 | 0.722 | 133.68ms |
| Pruned L1 (30%) | 0.501 | 0.698 | 86.42ms |
| Quantized INT8 | 0.500 | 0.697 | 71.10ms |
| **TensorRT FP16** | **0.528** | **0.718** | **21.06ms** |

> Latency figures reflect initial evaluation (cold-start). Live feed latency drops significantly post model warmup.

---

## �🔬 Optimization Deep-Dive

### 1. NMS-Free Transformer Architecture (RT-DETR)
Traditional object detectors use Non-Maximum Suppression (NMS) — a CPU-bound process that bottlenecks Edge AI pipelines. **RT-DETR** natively produces a fixed set of high-quality predictions. This keeps the entire pipeline on the **RTX 50-series GPU**, ensuring **deterministic latency** and zero post-processing lag.

### 2. Dynamic Token Scaling (Early Exit Innovation)
Inspired by "Early Exit" networks, we implement a confidence-aware resolution switcher in `early_exit.py`:
- **Stage 1 (320px):** Ultra-fast transformer encoder pass for simple frames.
- **Decision:** If the encoder is confident (> threshold), the system exits immediately.
- **Stage 2 (640px):** Escalates to full-precision decoding only for complex scenes.

This saves up to **70% of GPU cycles** on "easy" frames while maintaining peak accuracy when it matters.

### 3. Blackwell Hardware-Specific Tuning
The system is optimized for the **NVIDIA Blackwell (RTX 50-series)** architecture:
- **TF32 & BF16 AMP:** Leveraging Blackwell's Tensor Cores for 19-bit and 16-bit math, doubling throughput over FP32 with near-zero accuracy loss.
- **Dynamic VRAM Manager:** Automatically detects free system memory and sets a `per_process_memory_fraction` to maximize the CUDA memory manager's footprint without crashing the OS.
- **Expandable VRAM Segments:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces memory fragmentation for stable multi-model loading.

### 4. High-Precision Timing Rigor
To ensure optimizations are accurate, we use **Hardware-Synced Timing** via `torch.cuda.Event`. This eliminates CPU synchronization noise, providing sub-millisecond accurate tracking of actual GPU execution time.

---

## 🛡 Hardware Observability & Health
PrismNet provides a comprehensive "System Health" cockpit:
- **GPU Load Percentage:** Dynamic monitoring of Tensor Core utilization via native NVML (`nvidia-ml-py`), bypassing PyTorch readout bugs.
- **Memory Footprint:** Precise breakdown of process-specific vs. system-wide VRAM.
- **Predictive Health Report:** Real-time analysis of resource pressure.

---

## 🛠 Setup & Installation

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Export all model artifacts (skips if already present)
python3 main.py --mode export

# 4. Launch dashboard
python3 main.py --mode app
```

---

## 📊 Unified CLI & Benchmarking
The project features a single entry point (`main.py`) for all operations:

| Mode | Description |
|---|---|
| `--mode app` | Launches the Streamlit telemetry dashboard |
| `--mode export` | Generates all model artifacts (`.pt`, `.engine`) if missing |
| `--mode benchmark` | Runs a full comparison suite across all optimization tiers |
| `--mode coco-eval` | Evaluates model tiers against COCO val2017 metrics |
| `--mode baseline` | Tests the unoptimized FP32 transformer performance |
| `--mode early-exit` | Validates the dynamic resolution switching logic |

---

## 📂 File Structure
- `main.py`: Unified system entry point and CLI router.
- `st_app.py`: Modern minimalist Streamlit telemetry dashboard.
- `benchmark.py`: Data-driven performance analysis suite.
- `benchmark_coco.py`: COCO val2017 evaluation suite using `pycocotools`.
- `core/`: Model loader, dynamic resolution resolver, compressor, and config.
- `utils/`: Hardware telemetry (NVML GPU stats, camera detection, model size).
- `dashboard/`: Streamlit theme styles and badge renderer.

**Project PrismNet** — *The next generation of transformer-based edge vision.*
