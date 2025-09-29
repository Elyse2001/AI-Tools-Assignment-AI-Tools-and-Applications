import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("mnist_cnn.h5")

st.title("MNIST Digit Classifier")
uploaded = st.file_uploader("Upload a digit image (28x28)")

if uploaded:
    img = Image.open(uploaded).convert("L").resize((28,28))
    x = np.array(img)/255.0
    x = x.reshape(1,28,28,1)
    pred = model.predict(x).argmax()
    st.image(img, caption=f"Predicted digit: {pred}")
