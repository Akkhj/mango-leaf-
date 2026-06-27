import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Mango Leaf Disease Detector",
    page_icon="🥭",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("mango_leaf_model.h5")

with open("classes.json") as f:
    class_indices = json.load(f)

# classes.json maps label -> index
# Ensure label order matches model output index order.
labels = [None] * len(class_indices)
for label, idx in class_indices.items():
    labels[int(idx)] = label
labels = [l if l is not None else "Unknown" for l in labels]

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.title {
    font-size:40px;
    color:#2e7d32;
    text-align:center;
    font-weight:bold;
}
.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}
.result-box {
    padding:20px;
    border-radius:15px;
    background-color:#e8f5e9;
    text-align:center;
    font-size:20px;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🥭 Mango Leaf Disease Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload mango leaf image and detect disease using AI</div>", unsafe_allow_html=True)

# ---------------- UPLOAD IMAGE ----------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------- PREPROCESS ----------------
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------- PREDICTION ----------------
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    disease = labels[index]
    confidence = np.max(prediction) * 100

    # ---------------- RESULT ----------------
    st.markdown(f"""
    <div class="result-box">
    <h3>🌿 Disease: {disease}</h3>
    <h4>🎯 Confidence: {confidence:.2f}%</h4>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Please upload a leaf image to get prediction")
