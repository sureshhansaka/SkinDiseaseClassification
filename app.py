import os
import requests
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------------------------------------------
# Utility: Download model files from GitHub release if not available
# -------------------------------------------------------------------
def download_file(url, filename):
    if not os.path.exists(filename):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

# -------------------------------------------------------------------
# Load models once (cached)
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)

    MODEL1_URL = "https://github.com/sureshhansaka/SkinDiseaseClassification/releases/download/v1.0.0/best_resnet50_transfer_learning_model.h5"
    MODEL2_URL = "https://github.com/sureshhansaka/SkinDiseaseClassification/releases/download/v1.0.0/best_custom_cnn_attention_model.h5"

    model1_path = "models/model1.h5"
    model2_path = "models/model2.h5"

    download_file(MODEL1_URL, model1_path)
    download_file(MODEL2_URL, model2_path)

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)
    return model1, model2

model1, model2 = load_models()

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.title("🩺 Skin Disease Classification")
st.write("Upload an image and choose which deep learning model to use for prediction.")

# Sidebar
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model", 
    ["Model 1 - ResNet50 Transfer Learning", "Model 2 - Custom CNN + Attention"]
)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing (resize to 224x224 as used in training)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        if model_choice == "Model 1 - ResNet50 Transfer Learning":
            prediction = model1.predict(img_array)
        else:
            prediction = model2.predict(img_array)

        # Output prediction
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Prediction: Class {class_index} (Confidence: {confidence:.2f})")
