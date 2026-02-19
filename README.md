# CNN-Real-Time-Handwritten-Digit-Recognition
This project implements a deep learning-based Convolutional Neural Network (CNN) trained on the MNIST dataset for handwritten digit recognition. It extends the model to perform live digit detection through webcam input with real-time inference.

## 🚀 Project Overview

This project demonstrates:

* Training a CNN model on the MNIST dataset
* Saving the trained model (`.keras` format)
* Real-time handwritten digit detection using live camera feed
* Image preprocessing to match MNIST format (28x28 grayscale)
* Live prediction with confidence scores

## 🎥 Live Detection

The live detection script:

* Connects to mobile camera via streaming app (e.g., DroidCam / IP Webcam)
* Captures frames using OpenCV
* Converts frame to grayscale
* Resizes to 28x28
* Normalizes pixel values
* Feeds image to trained CNN model
* Displays predicted digit in real-time

Run: python live_detection.py

## 🏋️ Model Training

To train the model from scratch run : python model_training.py

After training, the model will be saved as: mnist_model.keras

## 🧰 Requirements

Typical dependencies:

* tensorflow
* keras
* opencv-python
* numpy
* matplotlib

(Exact versions inside `requirements.txt`)

## 🔍 How It Works

1. CNN extracts spatial features from digit images
2. MaxPooling reduces dimensionality
3. Dropout prevents overfitting
4. Dense layers perform classification
5. Softmax outputs probabilities for digits (0–9)

## 🎯 Key Highlights

✅ 99% Test Accuracy
✅ Real-time inference
✅ Mobile camera integration
✅ Lightweight & efficient architecture
✅ Clean modular structure
