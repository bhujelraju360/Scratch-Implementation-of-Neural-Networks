# Scratch Implementation of Neural Networks

This repository contains the **scratch (from scratch) implementation** of three fundamental neural network models using **only NumPy**, without using any deep learning libraries such as TensorFlow or PyTorch.  
This work is submitted as part of the **Applied Machine Learning (AML) assignment**.

---

## 📌 Implemented Models

### 1. Artificial Neural Network (ANN)
- Fully connected feedforward neural network
- One hidden layer
- ReLU activation function
- Softmax output layer
- Trained on a simple toy classification dataset
- Implements forward propagation and backpropagation manually

**File:** `ANN.py`

---

### 2. Convolutional Neural Network (CNN)
- Single convolution layer with 3×3 filter
- ReLU activation
- Flatten layer
- Fully connected output layer
- Uses a small toy image dataset for demonstration
- Convolution operation implemented manually using NumPy

**File:** `CNN.py`

---

### 3. Recurrent Neural Network (RNN)
- Simple vanilla RNN architecture
- Uses tanh activation function
- Processes sequential input data
- Demonstrates temporal dependency handling in sequences

**File:** `RNN.py`

---

## 🧪 Dataset Information

All models are trained and tested using **toy datasets** generated directly within the code:
- ANN: Synthetic 2D classification data
- CNN: Randomly generated small grayscale images
- RNN: Random numerical sequences

No external datasets are used.

---

## ⚙️ Requirements

- Python 3.x
- NumPy

Install NumPy if required:
```bash
pip install numpy
