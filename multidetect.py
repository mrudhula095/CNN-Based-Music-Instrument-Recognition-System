import os
import numpy as np
import tensorflow as tf
from preprocess import preprocess_audio
from segment import segment_audio

# ==================================================
# LABELS (MUST MATCH TRAINING ORDER EXACTLY)
# ==================================================
LABELS = [
    "brass",
    "flute",
    "guitar",
    "keyboard",
    "mallet",
    "reed",
    "string",
    "vocal"
]

# ==================================================
# FIXED MODEL PATH (WORKS LOCALLY + STREAMLIT CLOUD)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "instrunet_model_v3.keras")

# ==================================================
# LOAD MODEL
# ==================================================
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ==================================================
# SEGMENT-BASED PREDICTION
# ==================================================
def predict_segments(audio_path, visibility_threshold=0.1):
    """
    Segment-wise prediction with probability aggregation.

    Returns:
    - segment_predictions: list of dicts
    - instrument_summary: dict {instrument: avg_confidence}
    """

    model = load_model()

    segments, timestamps = segment_audio(audio_path)

    confidence_sum = {label: 0.0 for label in LABELS}
    segment_predictions = []

    for audio_chunk, (start, end) in zip(segments, timestamps):

        x = preprocess_audio_from_array(audio_chunk)
        preds = model.predict(x, verbose=0)[0]

        seg_result = {
            "start": float(start),
            "end": float(end),
            "predictions": {
                LABELS[i]: float(preds[i]) for i in range(len(LABELS))
            }
        }

        segment_predictions.append(seg_result)

        # Accumulate full probability distribution
        for i, label in enumerate(LABELS):
            confidence_sum[label] += preds[i]

    num_segments = len(segment_predictions)
    instrument_summary = {}

    for label in LABELS:
        avg_conf = confidence_sum[label] / num_segments
        if avg_conf >= visibility_threshold:
            instrument_summary[label] = float(avg_conf)

    return segment_predictions, instrument_summary


# ==================================================
# HELPER: PREPROCESS AUDIO FROM ARRAY
# ==================================================
def preprocess_audio_from_array(y, sr=22050):
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Time dimension normalization
    if mel_db.shape[1] < 128:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, 128 - mel_db.shape[1]))
        )
    else:
        mel_db = mel_db[:, :128]

    # Same normalization used during training
    mel_norm = (mel_db + 80.0) / 80.0
    mel_norm = mel_norm[..., np.newaxis]
    mel_norm = np.expand_dims(mel_norm, axis=0)

    return mel_norm