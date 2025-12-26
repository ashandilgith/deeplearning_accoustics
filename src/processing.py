import numpy as np
import os
import json
import tensorflow as tf
from src.preprocess import audio_to_spectrograms
from src.model import build_autoencoder

# Use a safe path for models
# On Hugging Face, /data is persistent. Locally, we use a folder named 'models'.
MODELS_DIR = "/data" if os.path.exists("/data") else "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def train_mode(audio_path, mode_name):
    """
    Trains an Autoencoder for a specific mode (Idle, Slow, or Fast).
    """
    # 1. Convert Audio to Training Data
    X_train = audio_to_spectrograms(audio_path)
    if X_train is None:
        return "Error: Audio file too short or invalid."

    # 2. Build & Train Model
    input_shape = X_train.shape[1:]
    autoencoder = build_autoencoder(input_shape)
    
    # We fit X to X (Input = Target) because it's an Autoencoder
    print(f"Training {mode_name} model...")
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=4, shuffle=True, verbose=0)

    # 3. Calculate Threshold (The limit of 'Normal')
    reconstructions = autoencoder.predict(X_train)
    mse = np.mean(np.power(X_train - reconstructions, 2), axis=(1, 2, 3))
    threshold = float(np.max(mse) * 1.1) 

    # 4. Save Model & Threshold
    model_path = os.path.join(MODELS_DIR, f"model_{mode_name}.h5")
    # include_optimizer=False makes the file smaller and safer to load
    autoencoder.save(model_path, save_format='h5', include_optimizer=False)
    
    # Save threshold
    meta_path = os.path.join(MODELS_DIR, f"meta_{mode_name}.json")
    with open(meta_path, 'w') as f:
        json.dump({"threshold": threshold}, f)
        
    return f"✅ Trained {mode_name.upper()}! Threshold set to {threshold:.5f}"

def predict_health(audio_path, mode_name):
    """
    Checks if new audio matches the learned pattern for the mode.
    """
    model_path = os.path.join(MODELS_DIR, f"model_{mode_name}.h5")
    meta_path = os.path.join(MODELS_DIR, f"meta_{mode_name}.json")
    
    if not os.path.exists(model_path):
        return "⚠️ Model not found. Please Train this mode first."

    # 1. Load Resources
    with open(meta_path, 'r') as f:
        threshold = json.load(f)["threshold"]

    # --- THE FIX IS HERE ---
    # compile=False prevents the 'mse' deserialization error
    model = tf.keras.models.load_model(model_path, compile=False)

    # 2. Process Input Audio
    X_test = audio_to_spectrograms(audio_path)
    if X_test is None:
        return "Error: Audio too short."

    # 3. Reconstruct & Measure Error
    reconstructions = model.predict(X_test)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=(1, 2, 3))

    # 4. Determine Health
    anomalies = np.sum(mse > threshold)
    total_slices = len(mse)
    
    health_score = 100 * (1 - (anomalies / total_slices))
    
    status = "🟢 HEALTHY" if health_score > 90 else "🔴 ANOMALY DETECTED"
    
    return f"""
    Result: {status}
    Mode: {mode_name.upper()}
    Health Score: {health_score:.1f}%
    (Anomalies found in {anomalies} of {total_slices} seconds analyzed)
    """