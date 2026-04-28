# 🚀 Neural Torque Estimator for PMSM/BLDC Motors

This project demonstrates a professional-grade **Deep Learning pipeline** for virtual sensing in industrial applications. The goal is to estimate electric motor torque based on high-frequency phase signals using a **Long Short-Term Memory (LSTM)** recurrent neural network, fully integrated into a **MATLAB/Simulink** simulation environment.

---

## 🛠 Tech Stack
*   **Deep Learning**: Python, PyTorch (Training & Weight Export)
*   **Simulation**: MATLAB, Simulink (Inference & Model-Based Design)
*   **Architecture**: Recurrent Neural Network (LSTM) optimized for time-series forecasting
*   **Hardware Target**: High-performance PMSM/BLDC motor controllers

---

## 📈 Project Architecture & Workflow

### 1. Model Development (PyTorch)
*   The model was trained in Python using a dataset of **60 electrical features**, including phase currents and control signals.
*   The architecture features **128 hidden units** to capture complex temporal dependencies and transient current spikes.
*   Weights were exported to a custom `.mat` format to ensure seamless integration without relying on external ONNX dependencies.

### 2. MATLAB/Simulink Integration
*   **Manual Network Assembly**: To bypass environment-specific compatibility issues, the network was reconstructed manually in MATLAB using the `dlnetwork` object.
*   **Real-Time Data Pipeline**: Utilizing `timeseries` objects for synchronized data streaming within the Simulink environment.
*   **Signal Calibration**: Implementation of post-processing blocks (Bias: `-0.07`, Gain: `20`) to align neural network outputs with physical units (N·m).

---

## 📂 File Structure

| File Name | Role |
| :--- | :--- |
| **`ProjectRem.slx`** | Core Simulink model with integrated Deep Learning Predict block. |
| **`run_me.mlx`** | Automation Live Script for workspace initialization and model assembly. |
| **`torque_lstm_model.mat`** | Compiled `dlnetwork` object ready for high-speed inference. |
| **`torque_model_Suc.pth`** | Verified PyTorch state dictionary containing optimized weights. |
| **`data_for_matlab.mat`** | Input dataset with 60 features (`X`) and ground truth (`Y`). |

---

## 📊 Results & Verification

The model successfully tracks transient current spikes and predicts corresponding torque values with high precision.

> **Visual Alignment**: The yellow dashed line represents the **LSTM prediction**, while the blue solid line shows the **reference signal (Ground Truth)**. After calibration (Bias: -0.07, Gain: 20), the model shows near-perfect alignment with actual signal dynamics.

---

## 🚀 How to Run
1.  Place all project files in your working directory (e.g., `C:\Users\ilaga\Documents\Wesa\`).
2.  Execute **`run_me.mlx`** to load parameters and assemble the network.
3.  Open **`ProjectRem.slx`** and set the simulation Stop Time to `79.4`.
4.  Run the simulation and open the **Scope** to verify the performance.
