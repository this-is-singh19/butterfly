import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TensorFlow model
model = tf.saved_model.load("saved_model.pb")

# Define the function to make predictions
def predict(image):
    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, [224, 224])
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Make prediction
    result = model(img_array)

    # Get the predicted label
    predicted_label = tf.argmax(result, axis=1).numpy()[0]

    return predicted_label

# Streamlit app
def main():
    st.title("Image Classification with Pretrained Model")

    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and make prediction
        image = Image.open(uploaded_image)
        label = predict(image)

        # Display the result label
        st.write(f"Prediction: Class {label}")

if __name__ == "__main__":
    main()
