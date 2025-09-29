import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load models once
@st.cache_resource
def load_models():
    model1 = load_model("https://github.com/sureshhansaka/SkinDiseaseClassification/releases/download/v1.0.0/best_resnet50_transfer_learning_model.h5")
    model2 = load_model("https://github.com/sureshhansaka/SkinDiseaseClassification/releases/download/v1.0.0/best_custom_cnn_attention_model.h5")
    return model1, model2

model1, model2 = load_models()

# Sidebar selection
st.sidebar.title("Deep Learning Models")
model_choice = st.sidebar.selectbox("Choose a model", ["Model 1 - Disease A", "Model 2 - Disease B"])

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing (adjust size to your model input)
    img_resized = img.resize((224, 224))  
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        if model_choice == "Model 1 - Disease A":
            prediction = model1.predict(img_array)
        else:
            prediction = model2.predict(img_array)

        st.success(f"Prediction: {np.argmax(prediction)} (Confidence: {np.max(prediction):.2f})")
