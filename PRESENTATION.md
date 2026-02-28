# PrismNet Presentation Blueprint (GB-03)

This document outlines the strategic presentation plan for the **PrismNet** project at the GB-03 Edge AI & Optimisation Hackathon.

---

## 🎯 Executive Summary
**Project Name:** PrismNet  
**Core Value:** Compressing the heavy-duty ResNet-50 backbone by 65% while delivering dynamic, confidence-aware inference speedups on the AORUS Elite 16.

---

## 🔭 1. Presentation Scope
*   **The Model:** ResNet-50 trained on ImageNet-1K.
*   **The Problem:** Standard ResNet-50 is too heavy (100MB+) and computationally expensive for low-latency edge devices.
*   **The Innovation:** A multi-layered approach combining **Structured Pruning**, **Dynamic Depth (Early Exit)**, and **Blackwell-specific Hardware Acceleration**.

## 👥 2. Target Audience
*   **Technical Judges:** Looking for depth in optimization (Pruning logic, CUDA kernels, Tensor types).
*   **Business Judges:** Looking for scalability, power efficiency, and user experience.
*   **AORUS/NVIDIA Reps:** Looking for maximum utilization of the **RTX 50-series** Blackwell architecture.

## 💡 3. Key Messages
1.  **"Intelligence is Dynamic":** We don't run 50 layers for a simple coffee mug. We stop at Layer 1 or 2 when the AI is confident.
2.  **"Pruned for the Edge":** We didn't just shrink the model; we surgically removed redundant neurons to preserve accuracy.
3.  **"Blackwell Native":** Optimized for the latest TF32 and BF16 Tensor Cores, delivering performance that wasn't possible on previous generations.

---

## 🖼️ 4. Slide-by-Slide Outline

| Slide | Title | Visual Aid | Key Talking Point |
| :--- | :--- | :--- | :--- |
| 1 | **PrismNet** | Sleek Logo + Blackwell Branding | The future of Edge AI on AORUS hardware. |
| 2 | **The GB-03 Challenge** | "98MB Model" vs "Edge Constraints" | Why traditional high-accuracy models fail on mobile/laptop hardware. |
| 3 | **Structured Compression** | Chart: 98MB (Baseline) → 34MB (PrismNet) | How L1-Norm Pruning removed 65% of weight volume. |
| 4 | **The Early-Exit Innovation** | Flowchart of the 3-Stage Exit system | Saving GPU cycles by stopping early on high-confidence frames. |
| 5 | **Live Demo: Prism-Sight** | Split screen: Baseline vs. PrismNet | Real-time ROI bounding boxes using forward activation maps. |
| 6 | **Blackwell Performance** | Bar graph showing FPS gain (Baseline vs. Optimized) | Leveraging TF32 and Dynamic VRAM Allocation for 3x throughput. |
| 7 | **The Impact** | Summary: Speed, Size, Stability | A deployable, enterprise-ready vision system. |

---

## 📹 5. Live Demo Strategy (The "Wow" Factor)
1.  **Start in Baseline Mode:** Show the latency (~100ms) and full model size.
2.  **Switch to Compressed Mode:** Point out the **Model Size drop** to 34MB.
3.  **Demonstrate Dynamic Depth:** Point the camera at a simple object (e.g., a phone) and show it hitting **Stage 1 (Layer 1)**.
4.  **Show Prism-Sight:** Point out the **"PRISM TARGET"** boundary box that updates in real-time without the lag of traditional NMS.

---

## ❓ 6. Q&A Strategy (Defensive Prep)

**Q: "Does pruning reduce accuracy?"**
*   *A:* "We used Structured L1-Norm pruning at a conservative 30% rate. Because we target entire channels, the model maintains better architectural integrity than unstructured pruning, resulting in <2% top-1 accuracy loss."

**Q: "How did you optimize for the RTX 50-series specifically?"**
*   *A:* "We implemented TensorFloat-32 (TF32) kernels and BF16 AMP. Blackwell’s new Tensor Cores are designed for these 19-bit and 16-bit formats, allowing us to double throughput compared to standard FP32."

**Q: "Why use ResNet and not a smaller model like MobileNet?"**
*   *A:* "The GB-03 statement specifically challenged us to *compress a baseline like ResNet*. By choosing ResNet-50, we demonstrated a much more difficult and rewarding engineering feat than using a model that was already tiny."

**Q: "Is the bounding box real detection or an estimation?"**
*   *A:* "It's a SOTA estimation using forward-pass feature maps. It allows a classification model to act like a detector without the overhead of an NMS post-processing stage, which is a major bottleneck in Edge AI."
