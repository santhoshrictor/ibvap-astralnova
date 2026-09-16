# IBVAP - Intelligent Boundary & Video Analytics Platform

An AI-powered video surveillance and activity analysis application built with **Streamlit**, **YOLOv8**, **OpenCV**, and **EasyOCR**.

## Features

- **Boundary & Perimeter Intrusion Monitoring**: Real-time fence line detection, perimeter cross detection, and alerting.
- **Activity & Pose Analysis**: Posture analysis and anomalous activity tracking via YOLOv8 Pose.
- **Vehicle & Plate Detection**: License plate recognition and vehicle monitoring with automatic alerting.
- **Animal Intrusion Tracking**: Automated identification and notification of animal presence in secured zones.
- **Interactive Streamlit Dashboard**: Live camera feeds, incident logging, and custom detection thresholds.

## Quickstart

### 1. Prerequisites
- Python 3.10+
- Recommended: GPU with CUDA support for accelerated model inference

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Application

```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit dashboard and UI logic.
- `activity_engine.py`: Computer vision processing, tracking, and intrusion detection engine.
- `alerting.py`: Alert dispatching system (webhooks, notifications, and logging).
- `*.pt` / `*.onnx`: Model weights for YOLO detection, pose estimation, and face detection.
