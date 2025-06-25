<!-- # CNVSRC2025
This Is Official CNVSRC2025 Competition Baseline System
<div align="center">
    <img src="example.gif" width="256" />
</div>

We now have two different tracks here:
1. Chinese VSR(Visual Speech Recognition) task, please see more information and code in [VSR file](https://github.com/liu12366262626/CNVSRC2025/tree/main/VSR)
2. Chinese VTS(Video to Speech) task, please see more information and code in [VTS file](https://github.com/liu12366262626/CNVSRC2025/tree/main/VTS) -->.


# CNVSRC2025: Official Baseline System

Welcome to the official baseline system for **CNVSRC2025**. This repository provides the starter code, pretrained models, and data processing tools for two major tracks in the competition: Visual Speech Recognition (VSR) and Video-to-Speech (VTS).

<div align="center">
    <img src="example.gif" width="256" />
</div>

---

## 🏁 About CNVSRC2025

CNVSRC2025 is a national-level challenge focused on **visual-based speech processing for Chinese**, aiming to promote advancements in lip-reading and silent speech generation technologies. Participants are encouraged to explore deep learning, multi-modal modeling, and generative methods in real-world noisy and unconstrained video scenarios.

---

## 📂 Available Tracks

### 🔹 1. Chinese Visual Speech Recognition (VSR)

This track requires models to **predict Chinese characters** from silent video input of a speaker's face. The main challenges include modeling subtle lip movements, handling visually similar phonemes, and dealing with variable lighting or occlusion.

🔗 **[Access VSR Track Code and Details](https://github.com/liu12366262626/CNVSRC2025/tree/main/VSR)**

Key components:
- Face and mouth ROI extraction
- Visual encoders (CNN, Transformer, Conformer, etc.)
- Sequence modeling with CTC loss
- Decoding and evaluation scripts (CER)

---

### 🔹 2. Chinese Video-to-Speech Synthesis (VTS)

In this track, participants must generate **intelligible and natural-sounding speech** from silent talking-head videos. The task focuses on prosody, articulation, and speaker similarity based on visual-only cues.

🔗 **[Access VTS Track Code and Details](https://github.com/liu12366262626/CNVSRC2025/tree/main/VTS)**

Key features:
- Video-based encoder extracting lip motion dynamics
- Diffusion-based or neural vocoder speech generation (e.g., HiFi-GAN)
- Support for waveform reconstruction and quality evaluation
- Evaluation metrics include STOI and CER testing

---

## 📦 Dataset & Evaluation

The competition uses Chinese audio-visual speech dataset, with tools for preprocessing, training, validation, and testing. The evaluation protocol includes:

- **For VSR**: Character Error Rate (CER)
- **For VTS**: Short-Time Objective Intelligibility (STOI), Character Error Rate (CER)

---

## 🚀 Getting Started

Each track includes a detailed `README.md` describing environment setup, dependencies, data preparation, training scripts, and evaluation methods. Please navigate to the respective folders for instructions:

```bash
cd VSR   # for Visual Speech Recognition
cd VTS   # for Video-to-Speech Synthesis
