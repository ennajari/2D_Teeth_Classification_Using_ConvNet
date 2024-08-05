import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from src.utils import predict_image
from src.model import create_model

# Charger le modèle (assurez-vous d'avoir un modèle entraîné)
model = create_model()
model.load_weights('path_to_your_trained_weights.h5')

st.title('Classification 2D des dents')

uploaded_file = st.file_uploader("Choisissez une image de dent...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée.', use_column_width=True)
    st.write("")
    st.write("Classification...")

    # Préparer l'image pour la prédiction
    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Faire la prédiction
    prediction = model.predict(img_array)
    
    # Afficher les résultats (ajustez selon vos classes)
    class_names = ['Classe 1', 'Classe 2']  # Remplacez par vos noms de classes réels
    st.write(f"Prédiction: {class_names[np.argmax(prediction)]}")
    st.write(f"Confiance: {np.max(prediction)*100:.2f}%")

st.write("Note: Ce modèle est un exemple et doit être entraîné avec vos données spécifiques.")