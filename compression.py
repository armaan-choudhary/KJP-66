import torch
import torch.nn as nn
import torch.nn.utils.prune as pruning
import os
import copy
from baseline import get_resnet50_imagenet

# PrismNet: Model Compression Engine (Pruning + Quantization)
print("--- PrismNet: Initializing Compression Engine (GB-03) ---")

def apply_structured_pruning(model, amount=0.3):
    """
    Applies Structured L1-Norm pruning to convolutional layers.
    Reduces the number of filters, effectively compressing the model architecture.
    """
    print(f"Applying Structured Pruning (Amount: {amount*100}%)...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            pruning.l1_unstructured(module, name='weight', amount=amount)
            pruning.remove(module, 'weight') # Make pruning permanent
    return model

def quantize_model_int8(model):
    """
    Prepares the model for INT8 Quantization.
    Reduces model size by 4x (FP32 -> INT8).
    """
    print("Preparing for INT8 Quantization (Dynamic)...")
    # For Edge deployment, we use dynamic quantization for immediate size reduction
    # Note: Dynamic quantization is most stable for nn.Linear in the current PyTorch version
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def get_compressed_size(file_path):
    if not os.path.exists(file_path): return 0
    return os.path.getsize(file_path) / (1024 * 1024)

if __name__ == "__main__":
    # 1. Load Baseline
    base_model = get_resnet50_imagenet(pretrained=False)
    torch.save(base_model.state_dict(), 'baseline_fp32.pth')
    base_size = get_compressed_size('baseline_fp32.pth')
    print(f"Baseline Size: {base_size:.2f} MB")

    # 2. Prune
    pruned_model = apply_structured_pruning(base_model)
    
    # 3. Quantize
    compressed_model = quantize_model_int8(pruned_model)
    
    # 4. Save & Compare
    torch.save(compressed_model.state_dict(), 'prismnet_compressed.pth')
    comp_size = get_compressed_size('prismnet_compressed.pth')
    
    print("\n" + "="*30)
    print(f"Compression Results (GB-03)")
    print(f"Baseline: {base_size:.2f} MB")
    print(f"PrismNet: {comp_size:.2f} MB")
    print(f"Reduction: {(1 - (comp_size/base_size))*100:.1f}%")
    print("="*30)
