# Real-Time Torque Estimation using LSTM and MATLAB/Simulink

This project demonstrates a Deep Learning approach to virtual sensing in industrial applications. The goal is to estimate the motor torque based on phase current signals using a Long Short-Term Memory (LSTM) neural network integrated into a Simulink simulation environment.

## 🚀 Overview
Predicting torque without physical sensors reduces cost and complexity in R&D and production. This project bridges the gap between high-level AI development (Python/PyTorch) and industrial system simulation (MATLAB/Simulink).

### Key Features:
- **Architecture**: LSTM-based Recurrent Neural Network for time-series forecasting.
- **Workflow**: Model trained in Python, exported to `.mat`, and integrated into Simulink.
- **Real-Time Integration**: Custom data pipeline using `timeseries` objects for streaming input.
- **Signal Calibration**: Post-processing blockset (Gain/Bias) to align neural network outputs with physical units.

## 🛠 Tech Stack
- **Deep Learning**: Python, PyTorch/TensorFlow (Training & Export)
- **Simulation**: MATLAB R2025b, Simulink
- **Toolboxes**: Deep Learning Toolbox, System Identification Toolbox

## 📊 Results
The model successfully tracks transient current spikes and predicts corresponding torque values with high precision.

[Insert your Scope Image here: e.g., images/final_scope_result.png]

The yellow dashed line represents the LSTM prediction, while the blue solid line shows the reference signal. After calibration (Gain: 12.0, Bias: -0.28), the model shows near-perfect alignment with actual signal dynamics.

## 📂 Project Structure
- `/notebooks`: Python training scripts and data preprocessing.
- `/models`: Exported `export_net.mat` file for Simulink.
- `/simulink`: `torque_prediction_model.slx` file.
- `/data`: Sample input current data for testing.

## 🚀 How to Run
1. Clone the repository.
2. Load `models/export_net.mat` into your MATLAB Workspace.
3. Open `simulink/torque_prediction_model.slx`.
4. Run the simulation to see real-time inference in the Scope block.

---
**Author**: Ilya Galyatin
**Focus**: R&D Engineer | Deep Learning | Industrial AI
