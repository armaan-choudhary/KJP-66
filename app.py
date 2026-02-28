import gradio as gr
import torch
import cv2
import time
import numpy as np
from baseline import get_resnet50_places365
from early_exit import EarlyExitResNet
import os

# PrismNet Project: Gradio UI + Live Webcam Feed
print("--- PrismNet: Launching Dashboard ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for models
baseline_model = None
optimised_model = None

def load_models():
    global baseline_model, optimised_model
    print("Loading PrismNet models into VRAM...")
    
    # Baseline (FP32)
    baseline_model = get_resnet50_places365(pretrained=False)
    
    # Optimised (Early-Exit)
    ee_model = EarlyExitResNet(baseline_model).to(device)
    # Load weights if they exist (from early_exit.py training)
    if os.path.exists('early_exit_model.pth'):
        try:
            ee_model.load_state_dict(torch.load('early_exit_model.pth', map_location=device))
            print("Loaded early_exit_model.pth weights.")
        except Exception as e:
            print(f"Could not load early_exit_model.pth: {e}")
            
    optimised_model = ee_model
    optimised_model.eval()
    baseline_model.eval()

def predict(frame, mode="Baseline"):
    """
    Inference function for Gradio.
    """
    if frame is None:
        return None, "No camera input detected."
    
    # Preprocessing
    try:
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).to(device).unsqueeze(0)
        
        start_time = time.time()
        
        with torch.no_grad():
            if mode == "Baseline":
                logits = baseline_model(img_tensor)
                exit_taken = "Final (FP32)"
            else:
                logits, exit_taken = optimised_model(img_tensor)
                exit_taken = f"Exit {exit_taken} (Dynamic)"
                
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            
        latency = (time.time() - start_time) * 1000
        
        # Result string with proper newline characters
        result_text = (
            f"PrismNet {mode} Mode\n"
            f"Prediction: Scene Class {pred.item()}\n"
            f"Confidence: {conf.item():.4f}\n"
            f"Inference Path: {exit_taken}\n"
            f"Latency: {latency:.2f} ms"
        )
        return frame, result_text
    except Exception as e:
        return frame, f"Error: {str(e)}"

def launch_ui():
    load_models()
    
    with gr.Blocks(theme=gr.themes.Soft(), title="PrismNet Scene Recognition") as demo:
        gr.Markdown("# PrismNet: On-Device Scene Recognition")
        gr.Markdown("### Optimized for AORUS Elite 16 (RTX 50-series)")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Using a simpler Gradio image streaming setup
                input_video = gr.Image(sources=["webcam"], label="Live Feed", streaming=True)
            with gr.Column(scale=1):
                mode_selector = gr.Radio(["Baseline", "Optimised"], label="Model Toggle", value="Baseline")
                output_text = gr.Textbox(label="Recognition Stats", lines=6)
                
        # Link input to prediction
        input_video.stream(fn=predict, inputs=[input_video, mode_selector], outputs=[input_video, output_text])
        
    print("Dashboard starting at http://127.0.0.1:7860")
    demo.launch(share=False) # share=False for stability in local dev

if __name__ == "__main__":
    launch_ui()
