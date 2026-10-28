# AARNet — Vision-Based Patient Monitoring & Abnormal Activity Detection

**Computer Vision | Deep Learning | Multi-Object Tracking | Healthcare AI | Automated Alerts**

## Project Overview

AARNet is an AI-powered patient monitoring research prototype designed to detect and track individuals in video streams, recognize potentially abnormal activities, and support automated emergency notifications.

The project combines **YOLOv8, DeepSORT, InsightFace, and spatiotemporal deep learning architectures** to explore identity-aware monitoring, fall detection, and abnormal activity recognition.

The system investigates how computer vision and temporal activity modeling can support safety monitoring in healthcare environments, particularly settings requiring continuous observation.
<img width="1781" height="464" alt="image" src="https://github.com/user-attachments/assets/76784fc4-d1c0-4ce4-8b0f-c7a8d3c08a9c" />

### Key Capabilities

- Real-time person detection using **YOLOv8**.
- Multi-object tracking and trajectory analysis using **DeepSORT**.
- Identity-aware tracking using **InsightFace facial embeddings**.
- Video-based activity recognition using **3D CNN and recurrent neural network architectures**.
- Fall-detection experiments using **MobileNetV2–GRU**.
- Automated alert demonstrations using **Twilio SMS**.
- Annotated video output and tracking metadata generation.
<img width="1496" height="722" alt="image" src="https://github.com/user-attachments/assets/78e9ec08-cd3b-43f9-9eb6-8ff0391fc5e8" />

## System Architecture
<img width="1416" height="697" alt="image" src="https://github.com/user-attachments/assets/be740ff8-7cc4-49ad-93c1-95386a85d9d9" />




The project consists of two complementary workflows.

### 1. Person Detection and Identity Tracking

**Video Input → YOLOv8 → DeepSORT → InsightFace → Annotated Tracking Output**

- YOLOv8 identifies people within video frames.
- DeepSORT maintains consistent tracking IDs across consecutive frames.
- InsightFace extracts facial embeddings for matching enrolled identities.
- Tracking history records individual movement trajectories.
- Annotated output displays bounding boxes, identities, track IDs, and FPS information.

### 2. Abnormal Activity Recognition and Alerting

**Video Input → Frame Preprocessing → Spatiotemporal Model → Activity Classification → Alert Logic → Twilio**

- Extracts and preprocesses video frame sequences.
- Uses deep learning architectures to identify activity patterns.
- Evaluates fall-related and abnormal activity classes.
- Applies detection thresholds to trigger warning events.
- Demonstrates automated notification workflows.

The components were developed and evaluated across separate scripts and notebooks. A fully integrated, clinically validated multi-camera deployment has not been established.
![Uploading image.png…]()


## Methodology

### 1. YOLOv8 — Person Detection

YOLOv8n is used to detect individuals in each video frame.

The detection pipeline:
- Processes video frames using OpenCV.
- Identifies person-class bounding boxes.
- Filters detections using confidence thresholds.
- Passes valid detections to the tracking module.

### 2. DeepSORT — Multi-Object Tracking

DeepSORT associates detections across consecutive video frames.

The tracking module:
- Assigns tracking IDs to detected individuals.
- Maintains short-term tracking histories.
- Records movement trajectories.
- Supports tracking visualization and JSON output.

### 3. InsightFace — Identity Recognition

InsightFace generates facial embeddings for enrolled individuals.

The system:
- Extracts facial representations.
- Compares embeddings using cosine similarity.
- Associates matched identities with tracking IDs.
- Labels unmatched individuals as unknown.

Identity recognition is an experimental component and should only be used with appropriate consent and privacy safeguards.

### 4. Spatiotemporal Activity Recognition

The project investigates several deep learning approaches for recognizing activities from video sequences.

**3D Convolutional Neural Networks**

Capture spatial and temporal information across consecutive video frames.

**3D CNN–LSTM**

Combines convolutional video feature extraction with recurrent temporal modeling to explore sequential activity patterns.

**MobileNetV2–GRU**

Uses lightweight visual feature extraction with gated recurrent units for sequence-based fall-detection experiments.

### 5. Automated Emergency Alerts

The fall-detection workflow demonstrates alerting mechanisms that can be activated when a model predicts a potential fall event.

Features include:
- Detection thresholds.
- On-screen warnings.
- Audible alerts.
- Twilio SMS notification integration.

Production emergency calling and independently benchmarked alert latency remain future validation requirements.

## Model Evaluation

### 3D CNN — Fall Classification

One of the fall-detection experiments produced the following results:

| Metric | Result |
|---|---:|
| Test samples | 111 |
| Accuracy | 65.77% |
| Positive-class precision | 90.48% |
| Positive-class recall | 34.55% |
| Positive-class F1-score | 0.5000 |

### Confusion Matrix

![Fall Detection Confusion Matrix](figures/fall_confusion_matrix.png)

The evaluation confusion matrix was:

| | Predicted Class 0 | Predicted Class 1 |
|---|---:|---:|
| Actual Class 0 | 54 | 2 |
| Actual Class 1 | 36 | 19 |

The model demonstrated high precision for positive predictions but missed a substantial number of positive-class examples.

This highlights the need to improve sensitivity before considering safety-critical deployment.

### Abnormal Activity Recognition

A separate abnormal-activity classification experiment reported a **ROC-AUC of 0.8850**.

This metric belongs to that specific experiment and does not represent verified performance of the entire patient-monitoring pipeline.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Computer Vision | OpenCV, YOLOv8 |
| Multi-Object Tracking | DeepSORT |
| Face Recognition | InsightFace |
| Deep Learning | TensorFlow, Keras |
| Neural Architectures | 3D CNN, LSTM, MobileNetV2, GRU |
| Data Processing | NumPy |
| Model Evaluation | Scikit-learn |
| Notifications | Twilio |
| Development | Jupyter Notebook |


## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MLAlchemy-cmd/AARNet-Intelligent-Patient-Monitoring.git
cd AARNet-Intelligent-Patient-Monitoring
```

### 2. Install Dependencies

```bash
pip install ultralytics deep-sort-realtime insightface opencv-python numpy tensorflow scikit-learn twilio
```

Additional dependencies and model weights may be required depending on the notebook and execution environment.

### 3. Prepare Tracking Inputs

Use an appropriately authorized webcam or video recording.

For identity recognition, generate local facial embeddings using `encode_known_faces.py` with consented example images.

### 4. Run the Tracking System

```bash
python src/subject_tracking.py
```

The script supports webcam and video-file processing.

### 5. Explore Activity Recognition

Open the notebooks to inspect video preprocessing, model architectures, training, evaluation, and fall-alert demonstrations.

Some notebooks require local datasets, trained model artifacts, and environment-specific configuration.

## Applications

Potential research applications include:

- Patient activity monitoring.
- Fall detection and safety monitoring.
- Multi-person movement analysis.
- Video-based abnormal activity recognition.
- Healthcare monitoring decision support.

This is an experimental research system and is not intended to replace clinical supervision or emergency-response procedures.

## Limitations and Future Improvements

- Improve fall-detection recall and reduce false negatives.
- Validate models across unseen subjects and recording environments.
- Measure tracking accuracy and identity-switch rates.
- Benchmark actual inference throughput and notification latency.
- Evaluate camera-angle and lighting robustness.
- Explore privacy-preserving tracking alternatives.
- Integrate event aggregation and escalation mechanisms.
- Conduct clinical validation before real-world healthcare deployment.

## Research and Recognition

**Research:** Real-Time Abnormal Activity Detection in Psychiatric Care Using Hybrid 3D CNN–LSTM and Identity Tracking.

**Recognition:** First Prize — National Hackathon.

[GitHub Profile](https://github.com/MLAlchemy-cmd)
