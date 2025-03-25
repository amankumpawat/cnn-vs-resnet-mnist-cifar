# deep-learning-classification-mnist-cifar10

A fully built-from-scratch deep learning project evaluating resnet-18, cnn, and no-skip variants on mnist and cifar-10. includes black-box and white-box testing, weight visualizations, hyperparameter tuning, and complete model analysis with formatted outputs and visuals.

### 🔍 End-to-End Training, Evaluation & Visualization using ResNet-18, CNN, and No-Skip Variants

This project was created for a senior capstone in deep learning. All models were built and trained **from scratch** using PyTorch and evaluated across multiple datasets. Visuals were carefully integrated to analyze training, predictions, weights, and testing phases.

---

## Models Implemented

- ✅ ResNet-18 (for MNIST)
- ✅ ResNet-18 (for CIFAR-10)
- ✅ Baseline CNN (no skip connections, CIFAR-10)
- ✅ ResNet-18 No-Skip (Ablation variant)

---

## Visualizations Included

- 📈 Training Accuracy & Loss (linear + log scale)
- 🔢 Confusion Matrices (all models)
- 🧠 Correct vs Incorrect Predictions
- 📉 Prediction Probability Bar Charts (softmax)
- 🎛 Model Summary Tables (torchsummary)
- 🎨 Conv1 Filters & FC Layer Weight Matrices
- 📋 Hyperparameter Tuning Tables (best/worst)

---

## Requirements

```bash
pip install torch torchvision matplotlib seaborn torchsummary
