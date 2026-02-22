# 👁️ Diabetic Retinopathy Detection Using EfficientNet on Fundus Images

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![AUC](https://img.shields.io/badge/Validation%20AUC-99.60%25-brightgreen?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-98.23%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**Core Stack:** Python (PyTorch, Flask) · EfficientNet-B0 · Transfer Learning · Grad-CAM · Medical Imaging

An automated deep learning system for **diabetic retinopathy severity grading** from fundus images, achieving **99.60% AUC** and **98.23% accuracy** across five clinical severity stages. Deployed as a Flask web application with integrated **Grad-CAM explainability** for transparent, interpretable predictions.

---

## 📋 Table of Contents
- [Results](#results)
- [How to Run](#how-to-run)
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Grad-CAM Explainability](#explainability-with-grad-cam)
- [Web Application](#web-application)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)

---

## Results

Strong validation performance across all five DR severity classes:

| Metric | Score |
|---|---|
| Validation AUC | **99.60%** |
| Validation Accuracy | **98.23%** |
| Validation F1 Score | **98.27%** |
| Best Epoch | Epoch 8 |

![Validation Results](images/diabetic_retinopathy_results.png)

AUC was selected as the primary metric due to its robustness against class imbalance and its clinical relevance in screening tasks.

---

## How to Run

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the repository

```bash
git clone https://github.com/egwaojeangel/diabetic_retinopathy_detection_using_efficientnet.git
cd diabetic_retinopathy_detection_using_efficientnet
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the APTOS 2019 dataset from Kaggle:
👉 [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

Place it in the root of the repository:

```
diabetic_retinopathy_detection/
├── dataset/
│   ├── train_images/
│   └── train.csv
├── images/
├── webapp.py
├── train.py
└── ...
```

### 5. Train the model

```bash
python train.py
```

### 6. Run the web application

```bash
python webapp.py
```

Then open your browser at: **http://127.0.0.1:5000**

### Demo Video

[![Demo Video](images/login_page.png)](https://drive.google.com/file/d/1juVdxAc2fzMCkKu3p3wP4xwWGuSdK_11/view?usp=sharing)

Click the thumbnail above to watch the full system demo.

---

## Overview

Diabetic retinopathy is one of the leading causes of preventable blindness worldwide. Manual grading of fundus images is time-intensive and requires trained ophthalmologists — a resource not always available in underserved regions.

This project demonstrates how deep learning can automate DR severity grading across five clinical stages, enabling large-scale screening programs and supporting clinical decision-making. The system goes beyond prediction accuracy by integrating Grad-CAM visualisations, making the model's decisions transparent and clinically interpretable.

### DR Severity Classes

| Grade | Stage |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## Dataset

**APTOS 2019 Blindness Detection** (Kaggle)

Color fundus images graded on a 0–4 DR severity scale by clinical experts.

| Split | Images |
|---|---|
| Training | 2,929 |
| Validation | 723 |
| **Total** | **3,652** |

Stratified splitting was applied to preserve class distribution across subsets.

> ⚠️ Dataset not included in this repository due to size and licensing.
> Download from: [Kaggle – APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

---

## Methodology

### Image Preprocessing
- Resized to **224 × 224 pixels** (EfficientNet-B0 input requirement)
- Converted to 3-channel RGB format
- ImageNet normalisation applied
- Tensor conversion via PyTorch transforms

### Data Augmentation (Training Only)
- Random horizontal flipping
- Random rotation
- Brightness and contrast adjustments

Augmentations applied to training set only to prevent contamination of validation performance.

---

## Model Architecture

**EfficientNet-B0** — pretrained on ImageNet, fine-tuned for 5-class DR grading.

EfficientNet achieves strong performance through compound scaling across three dimensions simultaneously:

- Network depth
- Network width  
- Input resolution

This gives it an excellent performance-to-parameter ratio, making it well-suited for medical imaging tasks where computational efficiency matters.

The final classification layer was replaced and fine-tuned to output 5 DR severity classes.

---

## Training Details

| Parameter | Value |
|---|---|
| Framework | PyTorch + timm |
| Loss Function | Cross-Entropy Loss |
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Epochs | 10 |
| Best Epoch | 8 |
| Checkpointing | Best model saved by validation AUC |

---

## Explainability with Grad-CAM

One of the key features that sets this system apart is integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** explainability.

### Why it matters

In medical AI, accuracy alone is not enough. Clinicians need to understand *why* a model made a prediction before trusting it. Grad-CAM shows exactly which regions of the fundus image drove the model's decision.

### How it works

1. Gradients are extracted from the final convolutional block
2. Class-specific importance weights are computed
3. A heatmap is generated and resized to match the input image
4. The heatmap is overlaid on the original fundus image using OpenCV
5. The result is displayed directly in the web interface

### Clinical relevance

Grad-CAM allows verification that the model is attending to clinically meaningful retinal features such as microaneurysms, haemorrhages, hard exudates, and neovascularisation — rather than irrelevant image artifacts.

---

## Web Application

A Flask-based web application provides a simple clinical screening interface:

- Upload a fundus image
- Real-time DR severity prediction
- Predicted class and confidence score displayed
- Grad-CAM heatmap visualisation shown alongside the result

Run locally with:

```bash
python webapp.py
```

---

## Requirements

```
torch
torchvision
timm
flask
flask-cors
numpy
Pillow
opencv-python
scikit-learn
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Limitations
- No external test set evaluation — metrics reflect internal validation only
- No cross-dataset generalisation study
- CPU-based training limited number of epochs
- No real-world clinical validation with expert ophthalmologists
- Not approved for clinical use

---

## Future Work
- External dataset validation for generalisation testing
- Class-weighted loss for better handling of class imbalance
- Larger EfficientNet variants (B3/B4) for potential performance gains
- Cloud deployment (Hugging Face Spaces / Render / AWS)
- Structured clinical report generation per patient
- Integration into a broader hospital AI diagnostic pipeline

---

## Ethical Considerations

This system is intended **strictly for research and educational purposes**. It is not validated for clinical diagnosis and should not replace professional ophthalmologic assessment. Real-world deployment would require regulatory approval and clinical validation.

---

## Disclaimer

This system is intended **strictly for research and educational use**. It is **not approved** for clinical diagnosis or treatment decisions. All outputs should be reviewed by qualified healthcare professionals.

---

## Author

**Angel Egwaoje**
Machine Learning Engineer | Computer Vision & Medical Imaging

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/angel-egwaoje-416927280)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/egwaojeangel)
