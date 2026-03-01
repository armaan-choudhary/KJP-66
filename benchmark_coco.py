import torch
import torchvision
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import time
import os
from datetime import datetime
import tabulate
from ultralytics import RTDETR

import core.config as cfg
from core.engine import get_rtdetr_engine
from utils.compression import load_quantized_state

def get_dataloader(data_dir, batch_size=1):
    val_dir = os.path.join(data_dir, "images", "val2017")
    ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
    
    if not os.path.exists(ann_file) or not os.path.exists(val_dir):
        print(f"Error: COCO val2017 not found at {data_dir}. Please download it.")
        return None, None
        
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((640, 640)),
        torchvision.transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.CocoDetection(root=val_dir, annFile=ann_file, transform=transform)
    
    # Custom collate to handle variable number of targets
    def collate_fn(batch):
        return tuple(zip(*batch))
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    return dataloader, ann_file

def evaluate_model(model_name, model_func, dataloader, device, is_trt=False):
    print(f"\n--- Evaluating {model_name} ---")
    results = []
    total_time = 0
    num_images = 0
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for images, targets in dataloader:
        if not targets[0]: # Skip images with no annotations
            continue
            
        img_id = targets[0][0]['image_id']
        orig_img_list = images
        
        # Batching logic (default batch_size=1)
        batch_imgs = torch.stack(list(images)).to(device)
        
        # Warmup and Timing
        start_event.record()
        
        with torch.no_grad():
            if is_trt:
                preds = model_func(batch_imgs)
            else:
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    preds = model_func(batch_imgs)
                    
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)
        num_images += len(images)
        
        # Process output format [cx, cy, w, h] to [x, y, w, h]
        out_boxes = []
        out_scores = []
        out_labels = []
        
        if is_trt:
            # Assuming TRT returns standard [B, 300, 84] tensor
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            if isinstance(preds, dict):
                # depends on trt engine export wrapper
                pass
                
            # Need to parse TRT output properly based on Ultralytics export
            # Simple fallback for now
            pass
        else:
            # Handle PyTorch model dict/tuple output
            if isinstance(preds, dict):
                logits = preds.get('pred_logits', None)
                boxes = preds.get('pred_boxes', None)
            elif isinstance(preds, (tuple, list)):
                if len(preds) > 0 and len(preds[0].shape) == 3 and preds[0].shape[-1] > 4:
                    boxes = preds[0][..., :4]
                    logits = preds[0][..., 4:]
                else: # Fallback based on RT-DETR specific structure
                    return None
                    
            if logits is not None and boxes is not None:
                probs = logits.softmax(-1)[..., :-1] # Drop background class
                scores, labels = probs.max(-1)
                
                # Format conversion for specific confident boxes (>0.001 to save space)
                for b_idx in range(len(boxes)):
                    valid_mask = scores[b_idx] > 0.001
                    v_boxes = boxes[b_idx][valid_mask]
                    v_scores = scores[b_idx][valid_mask]
                    v_labels = labels[b_idx][valid_mask]
                    
                    for i in range(len(v_boxes)):
                        cx, cy, w, h = v_boxes[i].cpu().tolist()
                        score = v_scores[i].item()
                        label = v_labels[i].item()
                        
                        # Scale back up to 80 COCO categories mapping if needed
                        # (simplification applied here. Usually RTDETR predicts 80 classes)
                        
                        # Convert to [x, y, w, h] absolute coordinates. Image is 640x640.
                        # Actually COCOeval expects original image scale, so we have to scale back.
                        # For baseline logic we will just dump raw relative or 640 scale, 
                        # but proper COCO eval requires matching the true image size.
                        
                        # Simplistic fallback using base inference to avoid manual bbox scaling logic:
                        pass
                        
    # For robust COCO eval using Ultralytics native validator:
    print(f"Calculated Latency: {total_time / num_images:.2f} ms per image.")
    return None, total_time / num_images

def validate_with_ultralytics(model_path, data_file="coco.yaml", imgsz=640, half=True, int8=False):
    # This uses the stable native validator to ensure correct bbox mapping and mAP metrics
    t0 = time.time()
    try:
        model = RTDETR(model_path)
        metrics = model.val(data=data_file, imgsz=imgsz, half=half, int8=int8, plots=False)
        t_lat = (time.time() - t0) * 1000 / 5000 # Appx fallback
        
        if hasattr(metrics, 'box'):
            map50 = metrics.box.map50
            map75 = metrics.box.map75
            map5095 = metrics.box.map
            
            # Fetch timing from metrics speed dict
            speed_ms = metrics.speed['inference'] if hasattr(metrics, 'speed') and 'inference' in metrics.speed else t_lat
            
            return {
                "mAP@0.5:0.95": map5095,
                "mAP@0.5": map50,
                "mAP@0.75": map75,
                "Latency_ms": speed_ms
            }
        else:
            return None
    except Exception as e:
        print(f"Validation failed for {model_path}: {e}")
        return None

def run_coco_eval(mode, data_dir):
    print("=======================================")
    print("   PRISMNET - COCO VAL2017 BENCHMARK   ")
    print("=======================================")
    
    # We will use ultralytics built-in validator which securely wraps pycocotools 
    # to avoid parsing the raw [cx, cy, w, h] logits manually for every engine type.
    
    results = {}
    
    models_to_run = []
    if mode in ["baseline", "all"]: models_to_run.append(("Baseline FP32", cfg.MODEL_BASE))
    if mode in ["pruned", "all"]: models_to_run.append(("Pruned L1 (30%)", cfg.MODEL_PRUNED))
    if mode in ["quantized", "all"]: models_to_run.append(("Quantized INT8", cfg.MODEL_QUANTIZED))
    if mode in ["distilled", "all"]: models_to_run.append(("Distilled Student", cfg.MODEL_DISTILLED))
    if mode in ["tensorrt", "all"]: models_to_run.append(("TensorRT Engine", cfg.MODEL_TRT))
    
    for name, path in models_to_run:
        print(f"\nEvaluating: {name} [{path}]")
        if not os.path.exists(path):
            print(f" -> Skipping: File {path} not found.")
            continue
            
        int8 = path == cfg.MODEL_QUANTIZED
        half = path == cfg.MODEL_TRT or path == cfg.MODEL_PRUNED
        metrics = validate_with_ultralytics(path, "coco.yaml", int8=int8, half=half)
        if metrics:
            results[name] = metrics
            print(f" -> mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
            print(f" -> Latency (Native):   {metrics['Latency_ms']:.2f} ms")
            
    # Print Table
    if results:
        headers = ["Tier", "mAP@0.5:0.95", "mAP@0.5", "mAP@0.75", "Latency (ms)"]
        table = []
        for name, metrics in results.items():
            table.append([name, 
                         f"{metrics['mAP@0.5:0.95']:.4f}", 
                         f"{metrics['mAP@0.5']:.4f}", 
                         f"{metrics['mAP@0.75']:.4f}", 
                         f"{metrics['Latency_ms']:.2f}"])
                         
        print("\n\n--- FINAL BENCHMARK RESULTS ---")
        print(tabulate.tabulate(table, headers=headers, tablefmt="grid"))
        
        # Save JSON
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"results/coco_eval_{mode}_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {out_file}")
    else:
        print("\nNo valid results generated.")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--data-dir", type=str, default="datasets/coco")
    args = parser.parse_args()
    run_coco_eval(args.mode, args.data_dir)
