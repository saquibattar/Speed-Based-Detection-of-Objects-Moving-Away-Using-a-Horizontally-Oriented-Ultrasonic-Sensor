# Speed-Based Detection of Objects Moving Away Using a Horizontally Oriented Ultrasonic Sensor

A machine learning project that detects whether an object is moving away from a single ultrasonic sensor and classifies the type of motion (medium, fast, or diagonal) based on speed and signal features.

---

## Author

**Saquib Attar**
Frankfurt University of Applied Sciences
saquib.attar@stud.fra-uas.de

---

## Project Overview

This project investigates whether a single horizontally mounted ultrasonic sensor (SRF02) can reliably detect away movement and estimate the speed of a moving object. Raw echo waveforms are collected using a Red Pitaya STEMlab board, converted to distance using the time-of-flight principle, cleaned with median filtering, and processed through a machine learning pipeline that classifies motion into three categories: **Medium**, **Fast**, and **Diagonal**.

---

## Pipeline Summary
---
Raw CSV Echo Data
      ↓
Echo Waveform Extraction (Column 18+)
      ↓
Time-of-Flight Distance Conversion
      ↓
Median Filtering (kernel size = 5)
      ↓
Stable Region Detection (Sliding Window, std < 0.02 m)
      ↓
Speed Estimation (v = displacement / duration)
      ↓
AWY Detection (positive slope = moving away)
      ↓
Feature Extraction (13 features per recording)
      ↓
ML Classification (DT / RF / KNN / SVM + LOO-CV)
      ↓
Output: Medium / Fast / Diagonal
---

---

## Hardware Setup

| Component | Details |
|---|---|
| Sensor | SRF02 Ultrasonic Range Finder |
| Measurement Board | Red Pitaya STEMlab (SoC, GNU/Linux) |
| Sampling Frequency | 1,953,125 Hz (echo waveform) |
| Frame Rate | 20 Hz |
| Minimum Sensing Distance | ~15 cm |
| Operating Voltage | 5V |
| Orientation | Horizontal, parallel to object path |

---

## Dataset

| Motion Type | Recordings | Use |
|---|---|---|
| Medium | 7 | Classification |
| Fast | 10 | Classification |
| Diagonal | 8 | Classification |
| Excluded | 1 | Did not match any class |
| **Total** | **25** | |

- Each recording is saved as a **CSV file**
- Each row = one measurement frame at 20 Hz
- Columns 1–17: header fields (sampling frequency, ADC settings, temperature, etc.)
- Column 18+: raw echo waveform (ADC samples)

---

## Extracted Features (13 per recording)

| Feature | Description |
|---|---|
| Start Distance | Stable distance reading before motion |
| End Distance | Stable distance reading after motion |
| Displacement | Difference between end and start distance |
| Duration | Time between motion start and motion end |
| Regression Speed | Slope of linear model: displacement / duration |
| Number of Dropouts | Count of frames with distance below 0.05 m |
| Dropout Ratio | Number of dropouts divided by total frames |
| Mean Amplitude | Average peak echo amplitude across all frames |
| Std Amplitude | Standard deviation of peak echo amplitude |
| Mean Echo Energy | Average std. deviation of echo waveform per frame |
| Std Echo Energy | Variation of echo energy across frames |
| Signal Range | Maximum minus minimum of filtered distance |
| Start Stability | Standard deviation of first 10 filtered frames |

---

## Machine Learning Models

All models trained and evaluated using **Leave-One-Out Cross-Validation (LOO-CV)** on 25 recordings.

| Model | Parameters | Accuracy | Macro F1 |
|---|---|---|---|
| Decision Tree | max_depth=3, random_state=42 | 76.0% | 0.763 |
| Random Forest | n_estimators=100, max_depth=3, random_state=42 | 80.0% | 0.798 |
| KNN | n_neighbors=3 | 76.0% | 0.762 |
| **SVM** | **kernel='rbf', random_state=42** | **88.0%** | **0.879** |

**Best model: SVM** with RBF kernel — 22/25 recordings correctly classified.

---

## Results — Confusion Matrices

![Confusion Matrices - All Models](images/ConfusionMatrix.png)

*Combined confusion matrices for all four classifiers (Decision Tree, Random Forest, KNN, SVM) under Leave-One-Out Cross-Validation. Darker blue = higher count. Diagonal cells = correct predictions.*

| Model | Accuracy | Diagonal ✓ | Fast ✓ | Medium ✓ |
|---|---|---|---|---|
| Decision Tree | 76.0% | 7/8 | 7/10 | 5/7 |
| Random Forest | 80.0% | 7/8 | 8/10 | 5/7 |
| KNN (k=3) | 76.0% | 5/8 | 8/10 | 6/7 |
| **SVM** | **88.0%** | **7/8** | **10/10** | **5/7** |

> **To display this image on GitHub:**
> 1. Create a folder called `images` in the root of your repository.
> 2. Place `ConfusionMatrix.png` inside that folder.
> 3. The image will render automatically when viewed on GitHub.

---

## Key Results

- **AWY detection rate: 100%** — every recording correctly identified as away movement
- **Best classifier: SVM** at 88.0% accuracy and macro F1-score of 0.879
- **Mean MAE: 0.408 m** | **Mean RMSE: 0.481 m** across all 25 recordings
- **Most important feature: End Distance** (importance score = 0.236) — primarily separates Diagonal class from Medium and Fast
- Main confusion occurs between **Fast and Medium** classes due to overlapping feature values

---

## Dependencies

pip install numpy pandas scipy scikit-learn matplotlib streamlit

| Library | Purpose |
|---|---|
| NumPy | Array operations and numerical computations |
| Pandas | Reading CSV files and organizing dataframes |
| SciPy (medfilt) | Median filtering for noise reduction |
| scikit-learn | ML models, cross-validation, scaling, evaluation |
| Matplotlib | Plotting confusion matrices and distance-time graphs |
| Streamlit | Interactive results dashboard |

---

## How to Run

1. Clone the repository and place your CSV recording files in the input folder.
2. Install dependencies:
   pip install numpy pandas scipy scikit-learn matplotlib streamlit
3. Run the processing pipeline:
   python pipeline.py --input_folder ./data
4. Launch the Streamlit dashboard:
   streamlit run dashboard.py

---

## Assumptions and Constraints

- Indoor lab environment only (Frankfurt University of Applied Sciences)
- Air temperature assumed at **20°C** for speed of sound calculation (c ≈ 343.4 m/s)
- Sensor minimum measurable distance: **~15 cm**
- Sensor orientation: **horizontal only**, beam aimed along object path
- Objects pushed **by hand** — no motorised or automated movement
- Dataset limited to **25 recordings** — results may not generalise beyond this setup
- Wall-read recordings (object exits sensor range) are automatically excluded

---

## Project Context

**Course:** Information Technology — Module: Machine Learning
**Supervisor:** Prof. Dr. Andreas Pech
**Institution:** Frankfurt University of Applied Sciences

---

## License

This project was developed for academic purposes as part of a university course. All rights reserved by the author.
