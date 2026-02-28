import torch
from ultralytics import RTDETR

def get_rtdetr_engine(model_path):
    """
    Loads RT-DETR with Blackwell-class optimizations.
    """
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RTDETR(model_path)
    model.to(device)
    return model
