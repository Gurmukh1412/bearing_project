# 🏭 Bearing Fault Diagnosis using Multimodal Fusion + Physics + Anomaly Detection

## 🚀 Overview

This project implements a **robust and interpretable bearing fault diagnosis system** by combining:

- 🔊 Vibration signals (raw time-series)
- ⚙️ Physics-based features (Envelope Spectrum / SCD)
- 🚨 Anomaly detection (Autoencoder)

These are fused using a **Cross-Modal Attention Network** for improved performance and reliability.

---

## 🧠 Key Features

- ✅ Multimodal fusion (Vibration + Physics + Anomaly)
- ✅ Physics-informed learning (Envelope spectrum)
- ✅ Anomaly detection for OOD scenarios
- ✅ Case-wise robustness (11 operating conditions)
- ✅ Real-time inference using Streamlit
- ✅ Interpretability via attention weights

---

## 📊 Evaluation Metrics

The system reports:

- Accuracy  
- Precision  
- Recall  
- Macro-F1 Score  

Additional analysis:

- Confusion Matrix  
- ROC Curve  
- Precision-Recall Curve  
- Confidence Distribution  
- Anomaly Score Distribution  

---

## 🏗️ Architecture


Raw Signal → Vib Encoder ┐
├──→ Fusion (Attention) → Classifier
Physics (SCD) → Encoder ┘

Anomaly → Autoencoder → Reconstruction Error


---

## 📁 Project Structure


bearing_project/
│
├── src/
│ ├── data/
│ │ └── loader.py
│ ├── models/
│ │ ├── fusion_model.py
│ │ └── scd_autoencoder.py
│ ├── train_fusion.py
│ ├── train_anomaly.py
│ └── evaluate_anomaly.py
│
├── models/
│ ├── fusion_model_best.pth
│ ├── scd_anomaly.pth
│ └── scaler.pkl
│
├── data/
│ └── (11 case folders with .mat files)
│
├── app.py
└── README.md


---

## ⚙️ Installation

```bash
git clone <repo_url>
cd bearing_project
pip install -r requirements.txt
▶️ Run the Dashboard
streamlit run app.py
🖥️ Dashboard Features
🔍 Input Modes
Dataset mode → select case (1–11)
Upload mode → upload .mat file
📊 Outputs
Prediction + Confidence
Anomaly Score
Raw Signal Plot
SCD Heatmap
Class Probabilities
📈 Evaluation
Confusion Matrix
ROC Curve
PR Curve
Confidence Distribution
Anomaly Score Distribution
🧠 Interpretability
Attention heatmap showing:
Vibration contribution
Physics contribution
🔧 Important Implementation Details
Fixed Signal Length

All signals are normalized to:

16384 samples

This ensures:

Stable Conv1D behavior
Consistent BatchNorm statistics
SCD Feature Extraction

Steps:

Bandpass filter
Hilbert transform → envelope
STFT
Log scaling
Resize to 64×64
🚨 Anomaly Detection
Autoencoder trained only on healthy data
Reconstruction error used as anomaly score
High error → Fault / abnormal
Low error → Normal
⚡ Performance Summary
Accuracy: ~0.80–0.90
Macro-F1: ~0.60–0.75
Low false positives
Fast inference (<10 ms)
🧠 Innovation
Physics + Deep Learning hybrid
Attention-based fusion
Joint classification + anomaly detection
Interpretable outputs
🚀 Future Work
Adaptive windowing
Online learning
Edge deployment
Multi-sensor fusion
🎯 Demo Strategy
Show performance across cases
Upload custom .mat file
Compare anomaly vs prediction
Explain attention heatmap
