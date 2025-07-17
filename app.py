import streamlit as st
from tensorflow.keras.models import load_model
from utils import preprocess_uploaded_image, make_gradcam_heatmap, overlay_gradcam_on_image
import cv2
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pneumonia Detector", layout="centered", page_icon="🩺")

import os
import requests

def download_model_from_drive():
    model_path = "pneumonia_cnn.h5"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            file_id = "1fyMPUuRuD2Ehs0MjQntKE7pVjSlsMRnI"  # Replace with your actual file ID
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            r = requests.get(download_url)
            with open(model_path, "wb") as f:
                f.write(r.content)

download_model_from_drive()


# --- LOAD MODEL ---
@st.cache_resource
def load_cnn_model():
    return load_model("../models/pneumonia_cnn.h5")

model = load_cnn_model()

# --- SIDEBAR INFO ---
with st.sidebar:
    st.title("🩻 Pneumonia Detection")
    st.markdown("""
    This AI model detects **pneumonia** from chest X-ray images using a CNN.  
    Built with **TensorFlow** and **Grad-CAM** for visual explainability.

    - Upload a grayscale chest X-ray.
    - Get prediction & confidence score.
    - See where the model is "looking".

    🔬 Trained on official **NIH chest X-ray** dataset.
    """)
    st.markdown("---")
    st.markdown("👨‍💻 Made by **Harsh**")

# --- HEADER ---
st.markdown("""
    <h1 style='text-align: center; color: #4b8bbe;'>🩺 Pneumonia Detector</h1>
    <p style='text-align: center;'>Upload a chest X-ray to predict <strong>Pneumonia</strong> and view Grad-CAM visualization.</p>
""", unsafe_allow_html=True)

# --- UPLOAD SECTION ---
uploaded_file = st.file_uploader("📤 Upload X-ray Image (PNG/JPG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.markdown("### 🔍 Preview")
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Original X-ray", use_container_width=True)

    # --- PROCESS IMAGE ---
    img_array, raw_image = preprocess_uploaded_image(uploaded_file)
    heatmap, prob = make_gradcam_heatmap(model, img_array)
    overlay = overlay_gradcam_on_image(raw_image, heatmap)

    with col2:
        st.image(overlay, caption="Grad-CAM Visualization", use_container_width=True)

    # --- PREDICTION ---
    result = "🦠 PNEUMONIA" if prob > 0.5 else "✅ NORMAL"
    confidence = prob if prob > 0.5 else 1 - prob

    st.markdown("---")
    st.markdown(f"<h3 style='text-align:center;'>Prediction: <span style='color:#e74c3c;'>{result}</span></h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align:center;'>Confidence: <code>{confidence:.2%}</code></h4>", unsafe_allow_html=True)
    st.markdown("---")

else:
    st.info("Please upload a grayscale chest X-ray image to begin.")

# --- FOOTER ---
st.markdown("""
    <div style='text-align: center; padding-top: 2em; font-size: 0.9em; color: gray;'>
        Built with ❤️ using TensorFlow & Streamlit
    </div>
""", unsafe_allow_html=True)
