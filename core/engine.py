import torch
from ultralytics import RTDETR

def get_rtdetr_engine(model_path):
    """
    Loads RT-DETR with specific precision defaults.
    """
    if torch.cuda.is_available():
        # Baseline: Force true FP32 math (No TF32)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.benchmark = True
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RTDETR(model_path)
    model.to(device)
    return model
