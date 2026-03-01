import torch
import torch.nn.utils.prune as prune
from ultralytics import RTDETR
import os
import core.config as cfg

def check_sparsity(model):
    """Calculates and prints model sparsity."""
    total_zeros = 0
    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            total_zeros += torch.sum(module.weight == 0).item()
            total_elements += module.weight.nelement()
    
    sparsity = 100. * total_zeros / total_elements
    print(f"Global Sparsity: {sparsity:.2f}%")
    return sparsity

def compress_rtdetr(source_path=cfg.MODEL_BASE, target_path="prismnet_compressed.pt", sparsity=0.3, run_tensorrt_export=False):
    """
    Executes optimization pipeline:
    1. L1 Unstructured Pruning
    2. Model quantization to INT8
    3. (Optional) TensorRT Compilation
    """
    print(f"\n--- Compression Pipeline ---")
    print(f"Loading baseline model: {source_path}")
    
    # Needs to be a fresh Ultralytics load to hook into the graph properly
    model = RTDETR(source_path)
    
    # 1. Pruning Stage
    print(f"\n[1/3] Applying L1 Unstructured Pruning (Ratio: {sparsity})...")
    
    modules_to_prune = []
    # Target linear and conv blocks
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            modules_to_prune.append((module, 'weight'))
        elif isinstance(module, torch.nn.Linear):
            modules_to_prune.append((module, 'weight'))
            
    # Apply global unstructured pruning
    prune.global_unstructured(
        modules_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    
    # Make sparsification permanent so it survives saving/loading
    for module, name in modules_to_prune:
        prune.remove(module, name)
        
    global_sparsity = check_sparsity(model.model)
    print(f"Global Sparsity: {global_sparsity:.2f}%")
    
    # 2. Quantization Stage
    print(f"\n[2/3] Applying INT8 Quantization...")
    # Custom quantization loop
    
    quantized_state = {}
    for name, param in model.model.state_dict().items():
        if param.is_floating_point():
            # Store at 8-bit scale
            scale = param.abs().max() / cfg.INT8_MAX_VAL
            if scale == 0: scale = 1.0
            q_v = torch.round(param / scale).to(torch.int8)
            quantized_state[name] = {'w': q_v, 's': scale}
        else:
            quantized_state[name] = param
            
    print(f"\n[3/3] Saving compressed model to {target_path}...")
    torch.save({'model': quantized_state}, target_path)
    
    base_size = os.path.getsize(source_path) / (1024 * 1024)
    opt_size = os.path.getsize(target_path) / (1024 * 1024)
    print(f"\n--- Compression Results ---")
    print(f"Baseline Size:   {base_size:.2f} MB")
    print(f"Compressed Size: {opt_size:.2f} MB")
    print(f"Total Reduction: {(1 - opt_size/base_size)*100:.2f}%\n")
    
    # 3. TensorRT Compilation Stage
    if run_tensorrt_export:
        print(f"\n--- TensorRT Pipeline ---")
        print(f"Compiling [{source_path}] to TensorRT Engine...")
        print(f"Targeting: {torch.cuda.get_device_name(0)}")
        print("This may take a few minutes.")
        try:
            trt_path = model.export(format='engine', half=True, dynamic=False, int8=False, imgsz=cfg.STAGE2_MAX_RES)
            print(f"TensorRT Engine successfully saved to: {trt_path}")
        except Exception as e:
            print(f"\nTensorRT Export Failed (Likely missing 'tensorrt' pip library natively): {e}")

if __name__ == "__main__":
    compress_rtdetr(run_tensorrt_export=True)
