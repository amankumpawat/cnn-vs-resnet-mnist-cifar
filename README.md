# Residual Networks for Robust Classification  
### A Hands-On Investigation of Vanishing Gradient Mitigation Across MNIST and CIFAR-10

## Repository Overview

This repository contains the code, visualizations, and evaluations for a deep learning project. The project explores convolutional neural networks (CNNs) and residual networks (ResNet-18) on the MNIST and CIFAR-10 datasets, with a focus on mitigating vanishing gradients and improving generalization.

The models are implemented in both **TensorFlow/Keras** and **PyTorch**, featuring robust training loops, performance monitoring, visualization tools, and structured internal/external documentation following the software engineering phases of analysis, design, coding, and testing.

---

## Project Objectives

- Compare standard CNNs and residual architectures (ResNet-18, No-Skip ResNet).
- Study the effects of batch normalization, dropout, and skip connections.
- Visualize learned features via convolutional filter maps and dense weights.
- Perform both **black-box** and **white-box testing** to validate results.
- Maintain clean modular design with internal documentation and traceability to project requirements.

---

## Notebook Sections & Cell Summary

### SECTION 1: Setup & Dependencies
- **Cell 1–2:** Mount Google Drive (optional) and install/import all necessary libraries (TensorFlow, PyTorch, scikit-learn, seaborn).

---

### SECTION 2: MNIST Dataset (TensorFlow/Keras)
- **Cell 3:** Load and preprocess MNIST, visualize class distribution, compute class weights.
- **Cell 4:** Define `MNIST_CNN_Improved` — a CNN with batch normalization, dropout, and multiple convolutional layers.
- **Cell 5:** Train the improved MNIST CNN model with progress monitoring.
- **Cell 6:** Evaluate performance on test data (accuracy, loss, confusion matrix).
- **Cell 7:** Plot example predictions and class probabilities (bar charts).
- **Cell 8:** Visualize learned conv1 filters and final dense layer weights.

---

### SECTION 3: CIFAR-10 Dataset (TensorFlow/Keras)
- **Cell 9:** Load CIFAR-10 data, visualize class balance, and compute class weights.
- **Cell 10:** Define `CIFAR10_CNN_Improved` — 3-block CNN with dropout, BN, and 256-dense final layer.
- **Cell 11:** Train the improved CIFAR-10 CNN model.
- **Cell 12:** Accuracy and loss plots (linear scale).
- **Cell 13:** Log-scale loss plot.
- **Cell 14:** Test set evaluation with confusion matrix.
- **Cell 15:** Visual predictions and bar plot probabilities.
- **Cell 16:** Conv and dense layer weight visualizations.

---

### 🔁 SECTION 4: ResNet Architectures (PyTorch)
- **Cell 17:** Define core ADTs:  
  - `ResidualBlock`,  
  - `ResNet18`,  
  - `BaselineCNN`,  
  - `ResNet18NoSkip`
- **Cell 18:** Define training loop (`train_and_evaluate_model`), loss functions (label smoothing), learning rate scheduling.
- **Cell 19:** Train ResNet-18 on MNIST (PyTorch) with SGD + Label Smoothing.
- **Cell 20:** Plot training and validation curves for MNIST ResNet.
- **Cell 21:** Confusion matrix and prediction plots (MNIST ResNet).
- **Cell 22:** Train ResNet-18 on CIFAR-10 (PyTorch).
- **Cell 23:** CIFAR ResNet training/validation curve visualization.
- **Cell 24:** Confusion matrix + predictions for CIFAR-10 ResNet.
- **Cell 25:** Train `ResNet18NoSkip` on CIFAR-10.
- **Cell 26:** Show loss/accuracy curves (no-skip variant).
- **Cell 27:** Train `BaselineCNN` on CIFAR-10 for ablation.
- **Cell 28:** Evaluate baseline CNN with confusion matrix.

---

### 📊 SECTION 5: Final Summaries & Testing
- **Cell 29:** Use `torchsummary` to show model summaries of all ResNet models.
- **Cell 30:** Visualize conv and dense layer weights across models for interpretability.
- **Cell 31:** Perform final **black-box testing** using `classification_report()` on all six models:
  - MNIST ResNet18  
  - CIFAR10 ResNet18  
  - CIFAR10 NoSkip  
  - CIFAR10 Baseline  
  - MNIST CNN (Keras)  
  - CIFAR10 CNN (Keras)

---

## Testing Strategy

### ✔ Black-Box Testing
- **Cell 31:** Runs `black_box_test()` to compare predictions vs true labels.
- Evaluation: Accuracy, precision, recall, and F1-score.

### ✔ White-Box Testing
- **Cells 8, 16, 20, 30:** Inspect model internals:
  - Training curves
  - Filter visualizations
  - Architecture summaries

---

## Models Implemented

| Model                 | Framework | Dataset   | Type               |
|----------------------|-----------|-----------|--------------------|
| CNN Improved          | TensorFlow| MNIST     | Deep CNN + BN/Dropout |
| CNN Improved          | TensorFlow| CIFAR-10  | Deep CNN + BN/Dropout |
| ResNet-18             | PyTorch   | MNIST     | Standard Residual  |
| ResNet-18             | PyTorch   | CIFAR-10  | Standard Residual  |
| ResNet-18 (No Skip)   | PyTorch   | CIFAR-10  | Ablation Variant   |
| Baseline CNN          | PyTorch   | CIFAR-10  | No-skip 4-block CNN|

---

## Visual Outputs

- Class distribution bar charts (MNIST & CIFAR)
- Training/validation loss + accuracy plots
- Confusion matrices
- Image prediction grids
- Bar charts of class probabilities
- Conv1 filter visualizations
- Dense layer heatmaps



