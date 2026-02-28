# PrismNet: AI Presentation Master Prompt & Notes

Use the following "Master Prompt" in any AI Slide Generator (e.g., Gamma.app, Tome.app, or ChatGPT with Canva) to create your deck.

---

## 🤖 Part 1: The Master Prompt (Copy & Paste)

**Theme:** Futuristic, Minimalist, High-Tech (Cyberpunk Dark Mode with Orange Accents)  
**Tone:** Professional, Visionary, Data-Driven  
**Objective:** Present "PrismNet" – a compressed ResNet-50 system for Edge AI (GB-03 Hackathon).

**Slide Content Structure:**
1.  **Title Slide:** PrismNet: SOTA Edge AI. Optimized for AORUS Elite 16 (NVIDIA Blackwell).
2.  **The Challenge:** The "ResNet Bloat" problem. High-accuracy models (100MB+) are too heavy for real-time edge devices.
3.  **Core Innovation 1: Structured Pruning:** How we used L1-Norm channel pruning to reduce model size by 65% (98MB → 34MB).
4.  **Core Innovation 2: Dynamic Depth (Early Exit):** Confidence-aware inference. Explain the 3-Stage system: Stage 1 (Turbo), Stage 2 (Mid), Stage 3 (Full).
5.  **Prism-Sight (Visual Localization):** Real-time ROI bounding boxes using forward activation maps – object detection features without the NMS overhead.
6.  **Hardware Performance:** Leveraging RTX 50-series (Blackwell). Metrics: TF32 acceleration, Dynamic VRAM allocation, and FPS gains.
7.  **Conclusion:** PrismNet is the blueprint for the next generation of mobile vision systems. Size reduced, speed doubled, accuracy preserved.

---

## 🎙️ Part 2: Speaker Notes (Slide-by-Slide)

### Slide 1: Title
*   "Good morning judges. I'm presenting PrismNet. We didn't just build an AI; we built an optimized engine designed specifically for the AORUS Elite 16 and the latest Blackwell architecture."

### Slide 2: The Challenge
*   "The GB-03 prompt asked for model compression. Why? Because a standard ResNet-50 is a 100MB beast. On a laptop or edge device, this leads to high latency and drained batteries. We set out to solve this 'Bloat' problem."

### Slide 3: Structured Pruning
*   "Our first layer of defense is Structured Pruning. Unlike unstructured methods that create sparse matrices, we surgically removed entire redundant filters. Result? A massive 65% reduction in weight volume while keeping the architecture dense and CUDA-friendly."

### Slide 4: Dynamic Depth (Early Exit)
*   "Innovation two is Intelligence on-demand. Not every image needs 50 layers of math. PrismNet is confidence-aware. For simple objects like a phone, it 'exits' at Stage 1. For complex scenes, it scales up. This saves up to 70% of GPU cycles per frame."

### Slide 5: Prism-Sight
*   "In our live demo, you'll see Prism-Sight. We implemented a way for a classification model to visually 'detect' objects using forward activation maps. This gives you YOLO-like boundaries without the heavy post-processing lag of traditional detectors."

### Slide 6: Blackwell Performance
*   "We optimized the CUDA kernels for the RTX 50-series. By enabling TF32 and dynamic VRAM allocation, we achieved a 3x throughput gain. We are squeezing every teraflop out of this Blackwell hardware."

### Slide 7: Conclusion
*   "PrismNet is smaller, faster, and smarter. It fulfills every requirement of the GB-03 track: Substantial size reduction, improved speed, and preserved accuracy. Thank you."

---

## 💡 Visual Aid Suggestions
- **Slide 3:** Use a bar chart showing the file size drop from 98MB to 34MB.
- **Slide 4:** Use a 3-step staircase graphic representing the "Exit Stages."
- **Slide 6:** Show a side-by-side FPS comparison: 12 FPS (Baseline) vs 45 FPS (PrismNet).
