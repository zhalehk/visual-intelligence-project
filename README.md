# Visual Intelligence Project — Binary Image Classification

A comparison of CNN and ScatNet architectures for binary image classification (cats vs dogs).

## Description

This project evaluates a custom CNN against a Scattering Network (ScatNet) on the cats vs dogs dataset, including explainability analysis using Grad-CAM.

## How to Run

Run the scripts in the following order:

1. **Train CNN**
   ```bash
   python src/train_cnn.py
   ```

2. **Train ScatNet**
   ```bash
   python src/train_scatnet.py
   ```

3. **XAI Analysis** (Grad-CAM visualizations)
   ```bash
   python src/xai_analysis.py
   ```

4. **Evaluate ScatNet**
   ```bash
   python src/evaluate_scatnet.py
   ```

## Results

| Model   | Test Accuracy |
|---------|--------------|
| CNN     | 96.92%       |
| ScatNet | 93.40%       |
