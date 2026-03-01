import torch
import os

# --- Model Configurations ---
MODEL_BASE = "rtdetr-x.pt"
MODEL_OPTIMIZED = "prismnet_compressed.pt"
MODEL_TRT = "rtdetr-x.engine"  # Hardware-Accelerated TensorRT Engine
COCO_CLASSES = 80 # Real class count for RT-DETR-X (Pretrained)

# --- Compression Settings ---
PRUNING_RATIO = 0.3
QUANTIZATION_DTYPE = 'INT8'
INT8_MAX_VAL = 127.0
MODEL_PRUNED = "prismnet_pruned.pt"
MODEL_QUANTIZED = "prismnet_compressed.pt"
MODEL_QUANTIZED = "prismnet_compressed.pt"

# --- Inference Settings ---
DEFAULT_THRESHOLD = 0.75
STAGE1_MIN_RES = 320
STAGE2_MAX_RES = 640 # Native RT-DETRimgsiz
DETECTION_CONF = 0.25
MODEL_STRIDE = 32 # Fetched from model.stride

# --- Hardware & System ---
GPU_FRACTION = 0.85
CAM_WIDTH = 1280
CAM_HEIGHT = 720
DEFAULT_CAM_ID = 0

# --- UI & Dashboard ---
UI_UPDATE_INTERVAL = 0.3
HISTORY_LIMIT = 100
RAM_CRITICAL_THRESHOLD = 85

# --- Precision Modes ---
PRECISION_BASELINE = 'highest'
PRECISION_OPTIMIZED = 'high'

# --- Project Identity ---
PROJECT_NAME = "PrismNet"
DASHBOARD_TITLE = "PrismNet SOTA"
DASHBOARD_ICON = "🧬"
