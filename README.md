# Pneumonia Detection Using Deep Learning

## Project Description

This project aims to develop a deep learning model
to detect pneumonia from chest X-ray images.

## Objectives

- Explore chest X-ray images.
- Build a CNN baseline model.
- Apply transfer learning.
- Evaluate model performance.
- Visualize predictions using Grad-CAM.

## Technologies

- Python
- PyTorch
- CNN
- ResNet18
- Scikit-learn
- Grad-CAM

## Project Structure

See the project directories for notebooks, source code,
datasets, models and reports.

## Installation

```bash
pip install -r requirements.txt

## Project Structure

```text
pneumonia-detection/
│
├── README.md
├── DATASET.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config.yaml
│
├── data/
│   ├── raw/
│   │   └── chest_xray/
│   │       ├── train/
│   │       │   ├── NORMAL/
│   │       │   └── PNEUMONIA/
│   │       ├── val/
│   │       │   ├── NORMAL/
│   │       │   └── PNEUMONIA/
│   │       └── test/
│   │           ├── NORMAL/
│   │           └── PNEUMONIA/
│   │
│   ├── processed/
│   │   ├── metadata.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── external/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_split.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_transfer_learning.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_gradcam.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── split.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── utils.py
│
├── scripts/
│   ├── download_data.sh
│   └── make_splits.py
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── models/
│   ├── best_model.pt
│   └── checkpoints/
│
├── logs/
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   │   └── results.csv
│   ├── rapport.pdf
│   └── presentation.pdf
│
└── tests/
    ├── test_dataset.py
    └── test_split_no_leak.py
```
