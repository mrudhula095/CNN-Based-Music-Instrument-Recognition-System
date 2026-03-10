import librosa
import numpy as np

SAMPLE_RATE = 22050
N_MELS = 128
FIXED_FRAMES = 128
MAX_DB = 80.0

def preprocess_audio(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS
    )

    # Power → dB
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Fix time dimension
    if mel_db.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)))
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]

    # Normalize to [0,1]
    mel_norm = (mel_db + MAX_DB) / MAX_DB

    # Shape for CNN
    mel_norm = mel_norm[..., np.newaxis]       # (128,128,1)
    mel_norm = np.expand_dims(mel_norm, axis=0)  # (1,128,128,1)

    return mel_norm