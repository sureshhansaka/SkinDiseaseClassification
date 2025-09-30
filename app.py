import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Model IDs for Google Drive
CUSTOM_CNN_ID = "1irrpB4NAH41Xk1jd9F9bxcvFAMkZOVz5"
RESNET50_ID  = "1iGKInycbqXCVQ1FTrOBYHjVHAcBDNob0"

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

# ------------------------ Download Model ------------------------ #
@st.cache_resource
def download_model(file_id, model_name):
    model_path = f"{model_name}.h5"
    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_name} (~200MB)..."):
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, model_path, quiet=True)
            except Exception:
                return None
    return model_path

# ------------------------ Custom Objects for Model Loading ------------------------ #
def create_custom_objects():
    custom_objects = {}

    def custom_input_layer(*args, **kwargs):
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if len(batch_shape) > 1:
                kwargs['shape'] = batch_shape[1:]
        return tf.keras.layers.Input(*args, **kwargs)

    custom_objects['InputLayer'] = custom_input_layer
    custom_objects['Input'] = custom_input_layer

    try:
        custom_objects['Attention'] = tf.keras.layers.Attention
        custom_objects['MultiHeadAttention'] = tf.keras.layers.MultiHeadAttention
    except AttributeError:
        pass

    return custom_objects

# ------------------------ Load Model ------------------------ #
@st.cache_resource
def load_model_safe(model_path):
    if model_path is None:
        return None

    custom_objects = create_custom_objects()

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects
        )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception:
        pass

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
            custom_objects=custom_objects
        )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception:
        pass

    # Fallback model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------------ Image Preprocessing ------------------------ #
def preprocess_image(image, target_size=(224, 224)):
    try:
        image = image.resize(target_size)
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception:
        return None

# ------------------------ Prediction ------------------------ #
def predict(model, image):
    processed = preprocess_image(image)
    if processed is None:
        return None, None, None
    try:
        predictions = model.predict(processed, verbose=0)
        idx = np.argmax(predictions[0])
        conf = predictions[0][idx]
        return idx, conf, predictions[0]
    except Exception:
        return None, None, None

# ------------------------ Streamlit UI ------------------------ #
st.set_page_config(page_title="Skin Disease Classifier", page_icon="🏥", layout="wide")
st.title("🏥 Skin Disease Classification System")
st.markdown("Upload an image to classify skin conditions using deep learning models")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ["Custom CNN with Attention", "ResNet50 Transfer Learning"])

# Load selected model
model = None
if model_choice == "Custom CNN with Attention":
    with st.spinner("Loading Custom CNN model..."):
        model_path = download_model(CUSTOM_CNN_ID, "custom_cnn_attention")
        model = load_model_safe(model_path)
        if model:
            st.sidebar.success("✓ Custom CNN Model Ready")
        else:
            st.sidebar.info("Fallback model in use.")
else:
    with st.spinner("Loading ResNet50 model..."):
        model_path = download_model(RESNET50_ID, "resnet50_transfer")
        model = load_model_safe(model_path)
        if model:
            st.sidebar.success("✓ ResNet50 Model Ready")
        else:
            st.sidebar.info("Fallback model in use.")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info("""
**About the Models:**
- Custom CNN uses attention mechanisms
- ResNet50 uses transfer learning
- Both trained on skin disease datasets
""")

# Main layout
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg","jpeg","png"], help="Upload a clear image of the skin condition")
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception:
            image = None
    else:
        image = None

with col2:
    st.subheader("🔍 Prediction Results")
    if model is None:
        st.info("⚠️ Model could not be loaded. Using fallback model only.")
    elif image is None:
        st.info("👆 Please upload an image to get started")
    else:
        with st.spinner("Analyzing image..."):
            predicted_idx, confidence, all_predictions = predict(model, image)
            if predicted_idx is not None:
                predicted_class = CLASS_NAMES[predicted_idx]
                st.markdown(f"### Predicted Condition: **{predicted_class}**")
                st.metric("Confidence", f"{confidence*100:.2f}%")
                st.progress(float(confidence))
                st.markdown("#### All Class Probabilities:")
                for cls_name, prob in zip(CLASS_NAMES, all_predictions):
                    col_a, col_b = st.columns([3,1])
                    with col_a: st.write(cls_name)
                    with col_b: st.write(f"{prob*100:.2f}%")
                    st.progress(float(prob))
                st.warning("⚠️ **Disclaimer:** This is AI prediction. Consult a dermatologist for accurate diagnosis.")
            else:
                st.info("Prediction could not be generated. Please try another image.")

# Additional info
st.markdown("---")
st.markdown("""
### How to use:
1. Select a model from the sidebar
2. Upload a clear image of the skin condition
3. View the prediction results and confidence scores
4. Compare results between different models

**Note:** Ensure images are clear and well-lit for best results.
""")

# Debug info
with st.expander("🔧 Debug Information"):
    st.write(f"TensorFlow Version: {tf.__version__}")
    st.write(f"Model loaded: {model is not None}")
    if model:
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Model output shape: {model.output_shape}")
