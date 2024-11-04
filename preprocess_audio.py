import os
import librosa
import numpy as np

# Define your dataset path
dataset_path = r"C:\Users\wasem\Downloads\RSDA Dataset-001\RSDA Dataset v5\Speaker_1"

def preprocess_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=22050)  # Default to 22050 Hz if no resampling is specified

    # Print the sample rate for each file
    print(f"Sample rate for {file_path}: {sr} Hz")
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Take the mean of the MFCCs to represent the audio file
    return np.mean(mfccs.T, axis=0)

# Load all audio files in the dataset path
data = []
labels = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):  # Adjust if your files have a different extension
            file_path = os.path.join(root, file)
            features = preprocess_audio(file_path)
            
            # Append the features and the label
            data.append(features)
            labels.append(root.split("\\")[-1])

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)
