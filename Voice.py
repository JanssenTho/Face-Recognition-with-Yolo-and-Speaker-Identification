import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf

def load_speaker_model(model_path):
    """Load the trained speaker identification model."""
    return tf.keras.models.load_model(model_path)

def extract_mfcc(audio, sample_rate, num_mfcc):
    """Extract MFCC features from the audio signal."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)  # Average over time to match model input
    return mfccs

def record_audio(duration, sample_rate):
    """Record audio using the microphone."""
    print("Recording...")
    audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()  # Wait for the recording to finish
    print("Recording complete.")
    return audio.flatten()  # Flatten 2D array to 1D

def predict_speaker(audio, model, label_to_speaker, sample_rate, num_mfcc):
    """Predict the speaker using the trained model."""
    mfccs = extract_mfcc(audio, sample_rate, num_mfcc)
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension for prediction
    predictions = model.predict(mfccs)
    predicted_label = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    predicted_speaker = label_to_speaker.get(predicted_label, "Unknown")
    return predicted_speaker, confidence
