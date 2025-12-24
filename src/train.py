import numpy as np
import tensorflow as tf
from src.model import build_model
import os

def train_dummy_model():
    print("--- 🛠️ Starting Piranaware Training Protocol ---")
    
    # 1. Create Fake Data (Simulating Spectrograms)
    # Shape: (100 samples, 128 height, 130 width, 1 channel)
    # This simulates 100 audio clips
    X_train = np.random.random((100, 128, 130, 1))
    
    # Create Fake Labels (0 = Healthy, 1 = Faulty)
    y_train = np.random.randint(2, size=(100,))

    # 2. Build the Brain
    input_shape = (128, 130, 1)
    model = build_model(input_shape)

    # 3. Train
    print("Training on synthetic data...")
    model.fit(X_train, y_train, epochs=5, batch_size=10)

    # 4. Save the Model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model.save('models/piranaware_brain.h5')
    print("--- ✅ Model Saved to models/piranaware_brain.h5 ---")

if __name__ == "__main__":
    train_dummy_model()