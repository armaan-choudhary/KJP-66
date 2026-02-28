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
            scale = v.abs().max() / 127.0
            if scale == 0: scale = 1.0
            q_v = torch.round(v / scale).to(torch.int8)
            quantized_dict[k] = {'w': q_v, 's': scale}
        else:
            quantized_dict[k] = v
    torch.save({'model': quantized_dict}, target_path)

def load_quantized_state(model_instance, source_path):
    """
    Dequantizes INT8 storage weights back to FP16 for runtime.
    """
    checkpoint = torch.load(source_path, map_location='cpu')
    quantized_state = checkpoint['model']
    fp16_state = {}
    for k, v in quantized_state.items():
        if isinstance(v, dict) and 's' in v:
            fp16_state[k] = (v['w'].to(torch.float32) * v['s']).to(torch.float16)
        else:
            fp16_state[k] = v
    model_instance.model.load_state_dict(fp16_state, strict=False)
    model_instance.model.half()
    return model_instance
