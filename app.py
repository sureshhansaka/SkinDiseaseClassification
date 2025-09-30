import io
import os
from typing import List, Tuple
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import gdown

# -----------------------
# CONFIGURATION
# -----------------------
# Google Drive file IDs
CUSTOM_CNN_ID = "1lw3jey8sgW2ALcEFeJiFHt2V6-mxJy-B"
RESNET50_ID = "1bpWH_mFMzT7_WpXUk_83svO9vaOKigpw"

# Model URLs
MODELS = {
    "Custom CNN with Attention": {
        "url": f"https://drive.google.com/uc?id={CUSTOM_CNN_ID}",
        "output": "custom_cnn_attention.h5",
        "description": "Custom CNN architecture with attention mechanisms"
    },
    "ResNet50 Transfer Learning": {
        "url": f"https://drive.google.com/uc?id={RESNET50_ID}",
        "output": "resnet50_transfer.h5",
        "description": "ResNet50 pre-trained with transfer learning"
    }
}

# Image processing parameters
IMG_SIZE = 224
TOPK = 5  # Show top 5 predictions

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
NUM_CLASSES = len(CLASS_NAMES)

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def download_model(url: str, output_path: str) -> str:
    """Download model from Google Drive if not already cached"""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    
    try:
        gdown.download(url, output_path, quiet=False)
        return output_path
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_model(path: str, model_name: str):
    """Load the trained model with custom layer support"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    try:
        # Try loading with custom attention layer for Custom CNN
        if "attention" in model_name.lower():
            from tensorflow.keras import layers
            
            class AttentionLayer(layers.Layer):
                def __init__(self, **kwargs):
                    super(AttentionLayer, self).__init__(**kwargs)
                
                def build(self, input_shape):
                    self.W = self.add_weight(
                        name='attention_weight',
                        shape=(input_shape[-1], 1),
                        initializer='random_normal',
                        trainable=True
                    )
                    self.b = self.add_weight(
                        name='attention_bias',
                        shape=(input_shape[1], 1),
                        initializer='zeros',
                        trainable=True
                    )
                    super(AttentionLayer, self).build(input_shape)
                
                def call(self, x):
                    e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
                    a = tf.keras.backend.softmax(e, axis=1)
                    output = x * a
                    return tf.keras.backend.sum(output, axis=1)
                
                def get_config(self):
                    return super(AttentionLayer, self).get_config()
            
            custom_objects = {'AttentionLayer': AttentionLayer}
            model = keras.models.load_model(path, custom_objects=custom_objects, compile=False)
        else:
            model = keras.models.load_model(path, compile=False)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    except Exception as e1:
        # Fallback: try loading without custom objects
        try:
            model = keras.models.load_model(path, compile=False)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e2:
            st.error(f"Failed to load model: {str(e2)[:300]}")
            raise

def resize_and_crop(im: Image.Image, target_size: int) -> Image.Image:
    """Resize image maintaining aspect ratio, then center crop"""
    w, h = im.size
    # Resize shorter side to slightly larger than target
    scale = max(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    im = im.resize((new_w, new_h), Image.BILINEAR)
    
    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    return im.crop((left, top, left + target_size, top + target_size))

def preprocess_image(pil_img: Image.Image, img_size: int) -> np.ndarray:
    """
    Preprocess image for model prediction
    Returns NHWC float32 array ready for model.predict
    """
    # Convert to RGB
    im = pil_img.convert("RGB")
    
    # Resize and crop
    im = resize_and_crop(im, img_size)
    
    # Convert to array and normalize
    arr = np.asarray(im, dtype=np.float32)  # HWC, 0..255
    arr = arr / 255.0  # Scale to [0, 1]
    
    return arr

def softmax_np(z: np.ndarray) -> np.ndarray:
    """Apply softmax to logits"""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z, dtype=np.float64)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

def predict_batch(model, pils: List[Image.Image]) -> np.ndarray:
    """Predict on a batch of images"""
    # Preprocess all images
    x = np.stack([preprocess_image(im, IMG_SIZE) for im in pils], 0).astype(np.float32)
    
    # Run prediction
    probs = model.predict(x, verbose=0)
    
    # Apply softmax if outputs are logits
    if probs.ndim == 2 and not np.allclose(probs.sum(axis=1, keepdims=True), 1.0, atol=1e-3):
        probs = softmax_np(probs)
    
    return probs

# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Skin Disease Classification System")
st.markdown("Upload skin lesion images to classify using deep learning models trained on dermatological datasets")

# Sidebar - Model Selection
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.radio(
    "Choose a model:",
    list(MODELS.keys()),
    help="Select which model to use for prediction"
)

# Display model info
selected_model_info = MODELS[model_choice]
st.sidebar.info(f"**{model_choice}**\n\n{selected_model_info['description']}")

# Download and load model
model_url = selected_model_info["url"]
model_output = selected_model_info["output"]

with st.spinner(f"Loading {model_choice}..."):
    model_path = download_model(model_url, model_output)
    
    if model_path:
        try:
            MODEL = load_model(model_path, model_choice)
            st.sidebar.success(f"✓ {model_choice} loaded successfully")
            
            # Test model output shape
            dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            out = MODEL.predict(dummy, verbose=0)
            
            if out.shape[-1] != NUM_CLASSES:
                st.sidebar.warning(
                    f"⚠️ Model outputs {out.shape[-1]} classes but expected {NUM_CLASSES}"
                )
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
    else:
        st.error("Could not download model. Please check your internet connection.")
        st.stop()

# Sidebar - Additional Info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.markdown(f"""
- **Input Size:** {IMG_SIZE}×{IMG_SIZE} pixels
- **Number of Classes:** {NUM_CLASSES}
- **Color Mode:** RGB
- **Preprocessing:** Resize & Center Crop
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.caption(
    "This application uses convolutional neural networks to classify "
    "various skin diseases and conditions from images."
)

# Main content
st.markdown("---")
st.markdown("### 📤 Upload Images")
st.caption("Upload one or more skin lesion images (JPG, PNG, or other common formats)")

uploaded_files = st.file_uploader(
    "Choose image files",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=True,
    help="Upload clear, well-lit images of skin conditions"
)

predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

# Prediction logic
if predict_button:
    if not uploaded_files:
        st.error("⚠️ Please upload at least one image before predicting.")
    else:
        pil_images, image_names = [], []
        
        # Load all uploaded images
        for uploaded_file in uploaded_files:
            try:
                pil_images.append(Image.open(io.BytesIO(uploaded_file.read())))
                image_names.append(uploaded_file.name)
            except Exception as e:
                st.warning(f"❌ Failed to read {uploaded_file.name}: {e}")
        
        if pil_images:
            with st.spinner("🔍 Analyzing images..."):
                probs = predict_batch(MODEL, pil_images)
            
            st.success(f"✓ Successfully analyzed {len(pil_images)} image(s)")
            st.markdown("---")
            
            # Display results for each image
            for i, pil_img in enumerate(pil_images):
                st.markdown(f"### 🖼️ Results for: `{image_names[i]}`")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(pil_img, caption=image_names[i], use_container_width=True)
                
                with col2:
                    if probs.shape[-1] != NUM_CLASSES:
                        st.error(
                            f"Model outputs {probs.shape[-1]} classes but "
                            f"expected {NUM_CLASSES}. Class mismatch!"
                        )
                        continue
                    
                    p = probs[i]
                    order = np.argsort(-p)[:TOPK]
                    
                    # Top prediction
                    top_class = CLASS_NAMES[int(order[0])]
                    top_confidence = p[int(order[0])] * 100
                    
                    st.markdown(f"#### Predicted Condition:")
                    st.markdown(f"### **{top_class}**")
                    st.metric("Confidence", f"{top_confidence:.2f}%")
                    st.progress(float(p[int(order[0])]))
                    
                    st.markdown("---")
                    st.markdown("#### Top Predictions:")
                    
                    for rank, k in enumerate(order, 1):
                        class_name = CLASS_NAMES[int(k)]
                        confidence = p[int(k)] * 100
                        
                        with st.container():
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"**{rank}. {class_name}**")
                            with col_b:
                                st.write(f"{confidence:.2f}%")
                            st.progress(float(p[int(k)]))
                
                st.markdown("---")
                st.warning(
                    "⚠️ **Medical Disclaimer:** This is an AI-powered tool for educational purposes only. "
                    "It should NOT replace professional medical diagnosis. Please consult a qualified "
                    "dermatologist for accurate diagnosis and treatment."
                )
                st.markdown("---")

# Footer information
st.markdown("---")
st.markdown("### 📖 How to Use")
st.markdown("""
1. **Select a Model** from the sidebar (Custom CNN or ResNet50)
2. **Upload Images** of skin lesions or conditions
3. **Click Predict** to analyze the images
4. **View Results** including top predictions and confidence scores
5. **Compare Models** by switching between them and re-predicting

**Tips for Best Results:**
- Use clear, well-lit images
- Ensure the skin condition is clearly visible
- Avoid images with excessive shadows or blur
- Center the lesion in the frame
""")

st.markdown("---")
st.markdown("### 🏷️ Classifiable Conditions")
with st.expander("View all conditions"):
    cols = st.columns(3)
    for idx, class_name in enumerate(CLASS_NAMES):
        with cols[idx % 3]:
            st.markdown(f"• {class_name}")

# Debug information
with st.expander("🔧 Technical Information"):
    st.markdown(f"""
    - **TensorFlow Version:** {tf.__version__}
    - **Model Loaded:** {model_choice}
    - **Model Path:** `{model_output}`
    - **Input Shape:** {MODEL.input_shape}
    - **Output Shape:** {MODEL.output_shape}
    - **Image Size:** {IMG_SIZE}×{IMG_SIZE}
    - **Number of Classes:** {NUM_CLASSES}
    """)
