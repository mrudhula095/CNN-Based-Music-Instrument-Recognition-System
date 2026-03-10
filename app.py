# ==============================
# InstruNet AI Streamlit App
# ==============================

import os
import warnings
import json
import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt

from preprocess import preprocess_audio
from multidetect import predict_segments
from harmonic_analysis import harmonic_analysis, estimate_perceived_age

warnings.filterwarnings("ignore")

# ==============================
# Silence TensorFlow logs
# ==============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ==============================
# LABEL ORDER (MATCH DATASET)
# ==============================
LABELS = ["bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed"]
CONDITION_LABELS = ["Healthy", "Aged", "Broken"]
PRESENCE_THRESHOLD = 0.15

# ==============================
# Streamlit Config
# ==============================
st.set_page_config(page_title="InstruNet AI", page_icon="🎧", layout="wide")

# ==============================
# CUSTOM UI STYLE
# ==============================
st.markdown("""
<style>
body {background:#020617;color:white;}
h1 {color:#22d3ee;text-align:center;}
.stButton>button {
    background:linear-gradient(45deg,#22c55e,#14b8a6);
    color:black;font-weight:bold;border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎶 InstruNet AI Audio Analyzer</h1>", unsafe_allow_html=True)
st.caption("Instrument Detection + Mel Spectrogram + JSON Export")

# ==============================
# BASE PATH
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_instrument_model():
    return tf.keras.models.load_model(os.path.join(BASE_DIR, "instrunet_model_v3.keras"))

@st.cache_resource
def load_condition_model():
    return tf.keras.models.load_model(os.path.join(BASE_DIR, "instrunet_model_final.keras"))

instrument_model = load_instrument_model()
condition_model = load_condition_model()

# ==============================
# INSTRUMENT PREDICTION
# ==============================
def predict_instrument(audio_path):
    x = preprocess_audio(audio_path)
    preds = instrument_model.predict(x, verbose=0)[0]
    idx = np.argmax(preds)
    return LABELS[idx], float(preds[idx]), preds

# ==============================
# CONDITION PREDICTION
# ==============================
def predict_condition(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    if len(y) < 16000 * 4:
        y = np.pad(y, (0, 16000*4 - len(y)))
    else:
        y = y[:16000*4]

    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_norm = (mel_db + 80) / 80
    mel_norm = mel_norm[..., np.newaxis]
    mel_norm = np.expand_dims(mel_norm, axis=0)

    preds = condition_model.predict(mel_norm, verbose=0)[0]
    idx = np.argmax(preds)

    return CONDITION_LABELS[idx], float(preds[idx])

# ==============================
# MEL SPECTROGRAM
# ==============================
def show_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

# ==============================
# WAVEFORM
# ==============================
def show_waveform(audio_path):
    y, sr = librosa.load(audio_path)
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Audio Waveform")
    st.pyplot(fig)

# ==============================
# JSON EXPORT FUNCTION
# ==============================
def export_json(audio_name, instrument, confidence, summary, condition):
    report = {
        "audio_file": audio_name,
        "predicted_instrument": instrument,
        "confidence": confidence,
        "instrument_summary": summary,
        "condition": condition
    }

    json_path = os.path.join(BASE_DIR, audio_name.replace(".wav", "_analysis.json"))

    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)

    return json_path

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    if st.button("🚀 Analyze Audio"):
        label, conf, _ = predict_instrument(temp_path)
        segments, summary = predict_segments(temp_path)
        condition, cond_conf = predict_condition(temp_path)

        st.success(f"🎵 Instrument: **{label.upper()}**")
        st.progress(conf)
        st.write(f"Confidence: `{conf:.3f}`")

        st.subheader("🧠 Condition")
        st.write(f"{condition} | Confidence: {cond_conf:.3f}")

        show_waveform(temp_path)
        show_spectrogram(temp_path)

        # Harmonic Analysis
        features = harmonic_analysis(temp_path)
        final_condition = estimate_perceived_age(features)

        st.subheader("🎚 Harmonic Features")
        st.json(features)
        st.write(f"Estimated Instrument Age: **{final_condition}**")

        # SAVE JSON FILE
        json_file = export_json(uploaded_file.name, label, conf, summary, condition)

        # DOWNLOAD BUTTON
        with open(json_file, "rb") as f:
            st.download_button("⬇ Download JSON Report", f, file_name=os.path.basename(json_file))

# ==============================
# FOOTER
# ==============================
st.markdown("<hr><center style='color:gray'>InstruNet AI | JSON Export Enabled</center>", unsafe_allow_html=True)
