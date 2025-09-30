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
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                st.success(f"✓ Downloaded {model_name}")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None
    
    return model_path

@st.cache_resource
def load_model_safe(model_path):
    """Load the trained model with error handling"""
    if model_path is None:
        return None
    
    try:
        # Try loading without compiling (avoids optimizer issues)
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompile with a basic optimizer
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        
        # Try alternative loading method
        try:
            with tf.keras.utils.custom_object_scope({}):
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                return model
        except Exception as e2:
            st.error(f"Alternative loading failed: {str(e2)}")
            return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Resize image
        image = image.resize(target_size)
        # Convert to array
        img_array = np.array(image)
        
        # Ensure RGB format
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

def predict(model, image):
    """Make prediction using the model"""
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, None, None
        
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return predicted_class_idx, confidence, predictions[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

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
model = None
if model_choice == "Custom CNN with Attention":
    with st.spinner("Loading Custom CNN model..."):
        model_path = download_model(CUSTOM_CNN_URL, "custom_cnn_attention")
        model = load_model_safe(model_path)
        if model is not None:
            st.sidebar.success("✓ Custom CNN Model Loaded")
        else:
            st.sidebar.error("❌ Failed to load Custom CNN Model")
else:
    with st.spinner("Loading ResNet50 model..."):
        model_path = download_model(RESNET50_URL, "resnet50_transfer")
        model = load_model_safe(model_path)
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
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            image = None
    else:
        image = None

with col2:
    st.subheader("🔍 Prediction Results")
    
    if model is None:
        st.error("⚠️ Model failed to load. Please try the following:")
        st.markdown("""
        1. Refresh the page
        2. Try the other model
        3. Check if models are accessible on GitHub
        """)
    elif image is None:
        st.info("👆 Please upload an image to get started")
    else:
        with st.spinner('Analyzing image...'):
            # Make prediction
            predicted_idx, confidence, all_predictions = predict(model, image)
            
            if predicted_idx is not None:
                predicted_class = CLASS_NAMES[predicted_idx]
                
                # Display main prediction
                st.markdown(f"### Predicted Condition: **{predicted_class}**")
                st.metric("Confidence", f"{confidence*100:.2f}%")
                
                # Display confidence bar
                st.progress(float(confidence))
                
                # Show all class probabilities
                st.markdown("#### All Class Probabilities:")
                for class_name, prob in zip(CLASS_NAMES, all_predictions):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"{class_name}")
                    with col_b:
                        st.write(f"{prob*100:.2f}%")
                    st.progress(float(prob))
                
                # Warning message
                st.warning("⚠️ **Disclaimer:** This is an AI prediction and should not replace professional medical diagnosis. Please consult a dermatologist for accurate diagnosis.")
            else:
                st.error("Failed to make prediction. Please try another image.")

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

# Debug information (hidden by default)
with st.expander("🔧 Debug Information"):
    st.write(f"TensorFlow Version: {tf.__version__}")
    st.write(f"Keras Version: {keras.__version__}")
    st.write(f"Model loaded: {model is not None}")
    if model is not None:
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Model output shape: {model.output_shape}")
