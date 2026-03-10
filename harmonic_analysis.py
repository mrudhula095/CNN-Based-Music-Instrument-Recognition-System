import librosa
import numpy as np

def harmonic_analysis(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Harmonic–percussive separation
    harmonic, noise = librosa.effects.hpss(y)

    # Energy-based Harmonic-to-Noise Ratio
    harmonic_energy = np.sum(harmonic ** 2)
    noise_energy = np.sum(noise ** 2) + 1e-9
    hnr = harmonic_energy / (harmonic_energy + noise_energy)

    # Spectral flatness (noise indicator)
    S = np.abs(librosa.stft(y))
    flatness = np.mean(librosa.feature.spectral_flatness(S=S))

    # Temporal stability (energy decay smoothness)
    rms = librosa.feature.rms(y=y)[0]
    decay_variance = np.var(rms)

    return {
        "harmonic_to_noise_ratio": float(hnr),
        "spectral_flatness": float(flatness),
        "decay_variance": float(decay_variance)
    }


def estimate_perceived_age(features):
    hnr = features["harmonic_to_noise_ratio"]
    flatness = features["spectral_flatness"]
    decay = features["decay_variance"]

    if hnr > 0.75 and flatness < 0.18 and decay < 0.005:
        return "New / Well-maintained"
    elif hnr > 0.55 and flatness < 0.3:
        return "Moderately Aged"
    else:
        return "Aged / Structurally Degraded"