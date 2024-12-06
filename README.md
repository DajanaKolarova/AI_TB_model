# Tuberculosis Detection Using Chest X-Ray Images

## Overview
This project focuses on using machine learning to detect tuberculosis (TB) from chest X-ray images. The goal is to build a model that can classify images into two categories: **healthy** and **tuberculosis-positive**. This project leverages a dataset of X-ray images from Kaggle and explores convolutional neural networks (CNNs) ResNet architectures to achieve reliable classification.

## Features
- Preprocessing pipeline for chest X-ray images.
- Binary classification: healthy vs tuberculosis positive.
- Implementation of ResNet.
- Visualizations of training progress and Grad-CAM for model interpretability.

## Dataset
The dataset contains X-ray images of lungs, divided into two classes:
- **Normal (healthy):** 700 images.
- **Tuberculosis-positive:** 700 images.
- original resolution: 512x512 pixels
  
**Source:** [Kaggle: Tuberculosis (TB) Chest X-Ray Database](https://www.kaggle.com/)

**Reduced dataset location**
https://drive.google.com/drive/folders/1hdM-e9RkgveMeWRY9jWjRKm3Mnm3lVKl?usp=share_link

## Preprocessing
1. Images are resized to `224x224` pixels.
2. Applied CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement.
3. Converted to grayscale for CLAHE and then reverted to RGB for model input.
4. Split into train, validation, and test sets with a ratio of 70:15:15.

## Model Architecture
1. **Baseline CNN**:
   - Custom convolutional neural network with two convolutional layers and two fully connected layers.
   - Relu activations and max pooling used for feature extraction.

2. **ResNet (Pre-trained)**:
   - Leveraged ResNet-18 with fine-tuning for binary classification.
   - Modified the final fully connected layer for two output classes.

## Training
- **Batch Size:** 32
- **Learning Rate:** 0.01
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Loss Function:** Cross-Entropy Loss
- **Number of Epochs:** 

## Results
The model achieved the following performance metrics on the test set:
- **Accuracy:**
- **Precision, Recall, and F1-Score**: Provided for each class in the classification report.

**Training vs Validation Loss:**
![Loss Curve](path-to-your-loss-curve-image)

**Confusion Matrix:**
- Class 0 (Healthy): High precision and recall.
- Class 1 (Tuberculosis): Some misclassifications noted but overall high accuracy.

