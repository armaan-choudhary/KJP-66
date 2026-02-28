import torch
from ultralytics import RTDETR
from core.config import PRECISION_BASELINE, MODEL_BASE
from utils.compression import load_quantized_state

def get_rtdetr_engine(model_path, is_compressed=False):
    """
    Loads RT-DETR. If compressed, loads the baseline then injects the INT8 state dict.
    """
    if torch.cuda.is_available():
        # Baseline: Force true FP32 math (No TF32)
        torch.set_float32_matmul_precision(PRECISION_BASELINE)
        torch.backends.cudnn.benchmark = True
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_compressed:
        # Load baseline architecture
        model = RTDETR(MODEL_BASE)
        # Load the FP32 inflated quantized dictionary
        model = load_quantized_state(model, model_path)
    else:
        model = RTDETR(model_path)
        
    if not str(model_path).endswith('.engine'):
        model.to(device)
        
    return model
