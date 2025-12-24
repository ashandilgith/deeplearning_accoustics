import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_shape):
    """
    Encoder: Compresses the audio image into a tiny latent vector.
    Decoder: Tries to recreate the original image from that vector.
    """
    # ENCODER
    input_img = layers.Input(shape=input_shape)
    
    # Conv1: Detect simple features (lines)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Conv2: Detect complex shapes
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x) # Bottleneck

    # DECODER
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x) # Expands size back up
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Output: Reconstructed Image
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse') # Mean Squared Error
    
    return autoencoder