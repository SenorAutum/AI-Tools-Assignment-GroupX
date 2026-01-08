import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model we trained in Task 2
model = tf.keras.models.load_model('mnist_model.h5')

st.title("MNIST Digit Classifier 🔢")
st.write("Upload an image of a handwritten digit (0-9) to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    img_array = np.array(image.resize((28, 28)))
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(img_array)
    label = np.argmax(prediction)
    
    st.success(f"Prediction: This is the number **{label}**")