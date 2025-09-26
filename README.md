# silent_disorder

Silent Speech EMG Signal Analysis and Classification
Project Overview
This project focuses on analyzing and classifying Electromyography (EMG) signals from silent speech data. The goal is to process raw EMG signals, extract meaningful features, and build a neural network model for classification tasks related to speech recognition from muscle activity.

Project Structure
The project follows a comprehensive pipeline for EMG signal processing and machine learning:

1. Data Loading and Exploration
Loads EMG data from text files containing time-series data

Parses and structures the data into proper time and EMG value columns

Initial data exploration through various visualization techniques

2. Data Visualization
The project includes multiple visualization approaches to understand the EMG signal characteristics:

Time Series Plot: Shows EMG values over time

Histogram: Displays frequency distribution of EMG values

Scatter Plot: Visualizes the relationship between time and EMG values

Spectrogram: Analyzes frequency content over time

3. Signal Preprocessing
Advanced signal processing techniques are applied:

Bandpass Filtering: Removes noise using a Butterworth filter (20-450 Hz)

Normalization: Standardizes EMG values using z-score normalization

Segmentation: Divides the signal into overlapping windows for analysis

4. Machine Learning Pipeline
Feature Engineering: Reshapes data for neural network input

Model Architecture: Implements a deep learning model with:

Input layer (1 feature)

Two hidden layers (32 neurons each, ReLU activation)

Output layer (2 classes, softmax activation)

Training: 30 epochs with batch size of 32

Evaluation: Performance assessment on test data

Key Features
Signal Processing: Professional-grade EMG signal filtering and normalization

Data Analysis: Comprehensive exploratory data analysis

Deep Learning: Neural network implementation using TensorFlow/Keras

Visualization: Multiple plotting techniques for data insight

Modular Code: Well-structured functions for reusability

Technical Specifications
Data Format
Input: Tab-separated text files with timestamp,EMG_value format

Sampling Frequency: 1000 Hz

Filter Parameters: 20-450 Hz bandpass filter

Model Architecture
text
Input Layer: 1 neuron
Hidden Layer 1: 32 neurons (ReLU)
Hidden Layer 2: 32 neurons (ReLU)
Output Layer: 2 neurons (Softmax)
Total Parameters: 1,186
Performance Metrics
Training Accuracy: ~50%

Validation Accuracy: ~46%

Loss Function: Categorical Crossentropy

Optimizer: Adam

Requirements
python
numpy
pandas
tensorflow/keras
scipy
matplotlib
scikit-learn
Usage
Place EMG data files in the appropriate directory structure

Update file paths in the code to match your data location

Run the script sequentially to:

Load and preprocess data

Visualize signals

Train the model

Evaluate performance

Potential Improvements
Implement more sophisticated feature extraction (MFCC, wavelet transforms)

Experiment with different neural network architectures (CNN, LSTM)

Add hyperparameter tuning

Incorporate cross-validation

Expand to multi-class classification

Add real-time prediction capabilities

Applications
This work has potential applications in:

Silent speech interfaces

Assistive technology for speech-impaired individuals

Biomedical signal processing

Human-computer interaction systems

Note
The current implementation uses randomly generated labels for demonstration purposes. In a real-world scenario, you would replace this with actual labeled EMG data corresponding to specific speech patterns or phonemes.

This project serves as a foundation for more advanced EMG-based speech recognition systems and demonstrates the complete pipeline from raw signal processing to machine learning model deployment.

