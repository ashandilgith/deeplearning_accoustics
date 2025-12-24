import librosa
import numpy as np

# Audio Settings
SAMPLE_RATE = 22050
DURATION = 1.0  # 1-second slices
SAMPLES_PER_SLICE = int(SAMPLE_RATE * DURATION)
N_MELS = 128

def audio_to_spectrograms(file_path):
    """
    1. Loads audio.
    2. Slices it into 1-second chunks.
    3. Converts each chunk to a Mel-Spectrogram (Image).
    Returns: Numpy array of shape (Num_Slices, 128, 44, 1)
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Calculate how many full 1-second slices we can get
        num_slices = len(y) // SAMPLES_PER_SLICE
        
        if num_slices < 1:
            return None # Audio too short

        spectrograms = []

        for i in range(num_slices):
            start = i * SAMPLES_PER_SLICE
            end = start + SAMPLES_PER_SLICE
            y_slice = y[start:end]

            # Create Mel Spectrogram
            spec = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=N_MELS)
            log_spec = librosa.power_to_db(spec, ref=np.max)
            
            # Normalize to 0-1 range (Neural Networks love 0-1)
            # We assume a dynamic range of 80dB for normalization
            norm_spec = (log_spec + 80) / 80
            norm_spec = np.clip(norm_spec, 0, 1)

            # Add Channel Dimension (Height, Width, Channel)
            norm_spec = norm_spec[..., np.newaxis]
            spectrograms.append(norm_spec)

        return np.array(spectrograms)

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None