import librosa
import numpy as np

SAMPLE_RATE = 22050

def segment_audio(
    file_path,
    segment_duration=2.0,
    hop_duration=1.0
):
    """
    Splits an audio file into overlapping segments.

    Returns:
    - segments: list of numpy arrays (audio chunks)
    - timestamps: list of (start_time, end_time)
    """

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    segment_samples = int(segment_duration * sr)
    hop_samples = int(hop_duration * sr)

    segments = []
    timestamps = []

    for start in range(0, len(y) - segment_samples + 1, hop_samples):
        end = start + segment_samples
        chunk = y[start:end]

        segments.append(chunk)

        start_time = start / sr
        end_time = end / sr
        timestamps.append((start_time, end_time))

    return segments, timestamps