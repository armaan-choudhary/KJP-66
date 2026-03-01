import torch
import os

def quantize_state_dict(model_instance, target_path):
    """
    Physically compresses state dict to INT8 for storage.
    """
    state_dict = model_instance.model.state_dict()
    quantized_dict = {}
    for k, v in state_dict.items():
        if v.is_floating_point():
            scale = v.abs().max() / cfg.INT8_MAX_VAL
            if scale == 0: scale = 1.0
            q_v = torch.round(v / scale).to(torch.int8)
            quantized_dict[k] = {'w': q_v, 's': scale}
        else:
            quantized_dict[k] = v
    torch.save({'model': quantized_dict}, target_path)

def load_quantized_state(model_instance, source_path):
    """Dequantizes INT8 storage weights back to FP32 for execution."""
    checkpoint = torch.load(source_path, map_location='cpu')
    quantized_state = checkpoint['model']
    fp32_state = {}
    
    for k, v in quantized_state.items():
        if isinstance(v, dict) and 's' in v:
            # Restore to FP32 and ensure contiguous memory for cuDNN
            fp32_state[k] = (v['w'].to(torch.float32) * v['s']).to(torch.float32).contiguous()
        else:
            if v.is_floating_point():
                fp32_state[k] = v.to(torch.float32).contiguous()
            else:
                fp32_state[k] = v.contiguous() if hasattr(v, 'contiguous') else v
                
    # Model remains in native Float32 precision
    model_instance.model.float()
    model_instance.model.load_state_dict(fp32_state, strict=False)
    
    return model_instance
