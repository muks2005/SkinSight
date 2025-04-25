import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#  MUST be the first Streamlit command
st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")

# === CONSTANTS ===
IMG_SIZE = 64
MODEL_PATH = "E:/SkinSight/ham10000_model.h5"
CSV_PATH = "E:/SkinSight/HAM10000_metadata.csv"

# === Load model and label encoder ===
@st.cache_resource
def load_model_and_labels():
    model = load_model(MODEL_PATH)

    df = pd.read_csv(CSV_PATH)
    le = LabelEncoder()
    le.fit(df['dx'])
    label_map = dict(zip(le.transform(le.classes_), le.classes_))

    return model, label_map

model, label_map = load_model_and_labels()

# === Streamlit UI ===
st.title(" Skin Lesion Classification App")
st.write("Upload a dermatoscopic image to classify the type of skin lesion using a trained MobileNet model.")

uploaded_file = st.file_uploader("Upload an image (JPG format)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    st.subheader("Model Prediction")
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    class_name = label_map[predicted_class]

    st.success(f" Predicted: **{class_name.upper()}** with **{confidence:.2f}%** confidence")

    # Show all class probabilities
    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({
        "Condition": [label_map[i] for i in range(len(prediction[0]))],
        "Probability (%)": prediction[0] * 100
    }).sort_values(by="Probability (%)", ascending=False)

    st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)
