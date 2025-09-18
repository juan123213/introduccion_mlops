import streamlit as st
import joblib
import numpy as np
from huggingface_hub import hf_hub_download # Librería para descargar desde el Hub

st.set_page_config(page_title="Consumidor de Modelos", layout="wide")

st.title('🌸 Aplicación de Predicción de Flores Iris')
st.write("Esta app consume un modelo entrenado automáticamente con GitHub Actions y publicado en Hugging Face.")

# --- Sección para descargar y cargar el modelo ---
st.header("1. Cargar el Modelo desde Hugging Face")

# El usuario debe introducir el ID de su repositorio en Hugging Face
repo_id_input = st.text_input(
    "Introduce el ID de tu Repositorio en Hugging Face (ej: tu-usuario/tu-repo):",
    "tu-usuario/mi-modelo-iris" # <-- Pon aquí tu repo como ejemplo
)

model = None
if st.button("Descargar y Cargar Modelo"):
    try:
        # Descargar el archivo model.pkl desde el Hub
        model_path = hf_hub_download(repo_id=repo_id_input, filename="model.pkl")
        
        # Cargar el modelo en la aplicación
        model = joblib.load(model_path)
        st.success(f"¡Modelo cargado exitosamente desde '{repo_id_input}'!")
        
        # Guardar el modelo en el estado de la sesión para no tener que descargarlo de nuevo
        st.session_state['model'] = model

    except Exception as e:
        st.error(f"No se pudo cargar el modelo. ¿El ID del repositorio es correcto y el archivo 'model.pkl' existe? Error: {e}")

# --- Sección para hacer predicciones ---
if 'model' in st.session_state and st.session_state['model'] is not None:
    st.header("2. Realizar una Predicción")
    
    st.sidebar.header('Introduce las Características de la Flor:')
    sepal_length = st.sidebar.slider('Largo del Sépalo (cm)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('Ancho del Sépalo (cm)', 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider('Largo del Pétalo (cm)', 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider('Ancho del Pétalo (cm)', 0.1, 2.5, 0.2)

    if st.button('Predecir Especie'):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        prediction = st.session_state['model'].predict(features)
        
        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_species = species_map.get(prediction[0], 'Desconocida')
        
        st.success(f"La especie predicha es: **{predicted_species}**")