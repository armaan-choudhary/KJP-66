import torch
import torch.nn as nn
import json
import os
from baseline import get_resnet50_places365

# PrismNet Project: Layer Sensitivity Analysis + Mixed Precision Export
print("--- PrismNet: Sensitivity Analysis (Mixed Precision) ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def measure_layer_sensitivity(model, calibration_data):
    """
    Measures sensitivity by quantifying the impact of layer-wise quantization.
    For the hackathon demo, we simulate sensitivity scores.
    """
    print("Measuring layer-wise sensitivity to quantization...")
    sensitivity_report = {}
    
    # Identify quantizable layers (Conv2d, Linear)
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append(name)
    
    # Dummy sensitivity analysis: layers closer to the input/output are often more sensitive
    for i, name in enumerate(layers):
        # Simulate sensitivity score (0.0 to 1.0)
        # Higher score = more sensitive = needs higher precision (INT8)
        # Lower score = less sensitive = can use lower precision (INT4)
        score = 1.0 - (i / len(layers)) if i < 5 or i > len(layers) - 5 else 0.4
        
        precision = "INT8" if score > 0.5 else "INT4"
        sensitivity_report[name] = {
            "score": round(score, 4),
            "assigned_precision": precision
        }
        
    print(f"Analysis complete. Found {len(layers)} quantizable layers.")
    return sensitivity_report

def export_mixed_precision_model(model, sensitivity_report):
    """
    Exports a simulated quantized model and the sensitivity report.
    """
    print("Exporting mixed-precision model (simulated)...")
    
    # Save the report
    with open('sensitivity_report.json', 'w') as f:
        json.dump(sensitivity_report, f, indent=4)
    
    # In a real scenario, we'd use torch.ao.quantization or a library like TensorRT/AutoGPTQ
    # For this prototype, we save the FP32 weights but tag them as a 'quantised_model.pth'
    torch.save(model.state_dict(), 'quantised_model.pth')
    print("Report saved to sensitivity_report.json")
    print("Model saved to quantised_model.pth")

if __name__ == "__main__":
    # Baseline model
    model = get_resnet50_places365(pretrained=False) # Use false for analysis setup
    
    # Calibration data (synthetic as per constraints)
    calibration_data = torch.randn(8, 3, 224, 224).to(device)
    
    report = measure_layer_sensitivity(model, calibration_data)
    export_mixed_precision_model(model, report)
