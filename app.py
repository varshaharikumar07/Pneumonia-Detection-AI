import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #E0E0E0;
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}
.result-box {
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.4rem;
    margin-top: 1rem;
    font-weight: 700;
}
.normal {
    background-color: #1B5E20;
    color: white;
}
.pneumonia {
    background-color: #B71C1C;
    color: white;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return keras.models.load_model("pneumonia_model_finetuned.keras")

model = load_model()
IMG_SIZE = (224, 224)

st.title("🫁 Pneumonia Detection using Xray")
st.write("Upload a chest X-ray image to check for Pneumonia Positive or Negative")

st.sidebar.header("Model Information")
st.sidebar.write("Model: Fine‑tuned ResNet50")
st.sidebar.write("Input size: 224×224")
st.sidebar.write("Threshold: 0.7")
st.sidebar.write("Performance: ~81% accuracy | ~96% recall")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    raw = model.predict(img_array, verbose=0)[0][0]
    pneumonia_score = raw
    normal_score = 1 - raw

    if pneumonia_score > 0.7:
        st.markdown(f"<div class='result-box pneumonia'> PNEUMONIA DETECTED</div>", unsafe_allow_html=True)
        st.caption(f"Pneumonia probability: {pneumonia_score:.2f}")
    else:
        st.markdown(f"<div class='result-box normal'> NORMAL</div>", unsafe_allow_html=True)
        st.caption(f"Normal probability: {normal_score:.2f}")
