import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

# Model URLs
CUSTOM_CNN_URL = "https://github.com/sureshhansaka/SkinDiseaseClassification/releases/download/v1.0.0/best_custom_cnn_attention_model.h5"
RESNET50_URL = "https://github.com/sureshhansaka/SkinDiseaseClassification/releases/download/v1.0.0/best_resnet50_transfer_learning_model.h5"

# Class names from your training dataset
CLASS_NAMES = [
    'Actinic keratosis', 
    'Atopic Dermatitis', 
    'Benign keratosis', 
    'Dermatofibroma', 
    'Melanocytic nevus', 
    'Melanoma', 
    'Squamous cell carcinoma', 
    'Tinea Ringworm Candidiasis', 
    'Vascular lesion'
]

@st.cache_resource
def download_model(url, model_name):
    """Download model from GitHub releases"""
    model_path = f"{model_name}.h5"
    
    if not os.path.exists(model_path):
        with st.spinner(f'Downloading {model_name}...'):
            response = requests.get(url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
    
    return model_path

@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    try:
        # Try loading with compile=False to avoid optimizer issues
        return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Try with custom objects if needed
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e2:
            st.error(f"Failed to load model: {e2}")
            return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Resize image
    image = image.resize(target_size)
    # Convert to array
    img_array = np.array(image)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image):
    """Make prediction using the model"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return predicted_class_idx, confidence, predictions[0]

# Streamlit UI
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Skin Disease Classification System")
st.markdown("Upload an image to classify skin conditions using deep learning models")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose a model:",
    ["Custom CNN with Attention", "ResNet50 Transfer Learning"]
)

# Load selected model
if model_choice == "Custom CNN with Attention":
    model_path = download_model(CUSTOM_CNN_URL, "custom_cnn_attention")
    model = load_model(model_path)
    if model is not None:
        st.sidebar.success("✓ Custom CNN Model Loaded")
    else:
        st.sidebar.error("❌ Failed to load Custom CNN Model")
else:
    model_path = download_model(RESNET50_URL, "resnet50_transfer")
    model = load_model(model_path)
    if model is not None:
        st.sidebar.success("✓ ResNet50 Model Loaded")
    else:
        st.sidebar.error("❌ Failed to load ResNet50 Model")

# Display model info
st.sidebar.markdown("---")
st.sidebar.info("""
**About the Models:**
- Custom CNN uses attention mechanisms
- ResNet50 uses transfer learning
- Both trained on skin disease datasets
""")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a skin image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the skin condition"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🔍 Prediction Results")
    
    if uploaded_file is not None:
        with st.spinner('Analyzing image...'):
            # Make prediction
            predicted_idx, confidence, all_predictions = predict(model, image)
            predicted_class = CLASS_NAMES[predicted_idx]
            
            # Display main prediction
            st.markdown(f"### Predicted Condition: **{predicted_class}**")
            st.metric("Confidence", f"{confidence*100:.2f}%")
            
            # Display confidence bar
            st.progress(float(confidence))
            
            # Show all class probabilities
            st.markdown("#### All Class Probabilities:")
            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, all_predictions)):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"{class_name}")
                with col_b:
                    st.write(f"{prob*100:.2f}%")
                st.progress(float(prob))
            
            # Warning message
            st.warning("⚠️ **Disclaimer:** This is an AI prediction and should not replace professional medical diagnosis. Please consult a dermatologist for accurate diagnosis.")
    else:
        st.info("👆 Please upload an image to get started")

# Additional information
st.markdown("---")
st.markdown("""
### How to use:
1. Select a model from the sidebar
2. Upload a clear image of the skin condition
3. View the prediction results and confidence scores
4. Compare results between different models

**Note:** Ensure images are clear and well-lit for best results.
""")
