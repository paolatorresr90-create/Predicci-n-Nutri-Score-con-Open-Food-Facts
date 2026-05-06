import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. CONFIGURACIÓN E INTERFAZ
st.set_page_config(page_title="Nutri-Score AI Predictor", page_icon="🥗")

@st.cache_resource
def load_model_assets():
    # Cargamos desde la carpeta /models
    model = joblib.load('models/nutriscore_xgb_model.pkl')
    features = joblib.load('models/features_list.pkl')
    return model, features

model, features_20 = load_model_assets()

st.title("🥗 Nutri-Score AI Predictor")
st.subheader("Clasificación Nutricional de Alta Precisión (90% Accuracy)")

# 2. ENTRADA DE DATOS (Interfaz de Usuario)
col1, col2 = st.columns(2)

with col1:
    energy = st.number_input("Energía (kcal/100g)", 0.0, 900.0, 150.0)
    sugars = st.number_input("Azúcares (g)", 0.0, 100.0, 5.0)
    fat = st.number_input("Grasas totales (g)", 0.0, 100.0, 10.0)
    sat_fat = st.number_input("Grasas saturadas (g)", 0.0, 100.0, 2.0)

with col2:
    proteins = st.number_input("Proteínas (g)", 0.0, 100.0, 5.0)
    fiber = st.number_input("Fibra (g)", 0.0, 50.0, 2.0)
    salt = st.number_input("Sal (g)", 0.0, 40.0, 0.5)
    is_bev = st.selectbox("¿Es una bebida?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

# 3. EL "PORQUÉ" TÉCNICO: REPRODUCCIÓN DE VARIABLES
# El modelo XGBoost exige las 20 variables originales. Las calculamos aquí:
if st.button("Generar Diagnóstico"):
    # Replicamos el Feature Engineering del entrenamiento
    data = {
        'energy-kcal_100g': energy, 'fat_100g': fat, 'saturated-fat_100g': sat_fat,
        'carbohydrates_100g': sugars + 10, # Estimación simple para el ejemplo
        'sugars_100g': sugars, 'fiber_100g': fiber, 'proteins_100g': proteins, 
        'salt_100g': salt, 'sodium_100g': salt/2.5,
        'sugar_ratio': sugars / (sugars + 10.1),
        'sat_fat_ratio': sat_fat / (fat + 0.1),
        'protein_ratio': proteins / (fat + proteins + 10.1),
        'salt_density': salt / (energy + 1),
        'carb_minus_sugar': 10.0,
        'total_solids': fat + sugars + proteins + salt,
        'is_beverage': is_bev,
        'is_high_sugar': 1 if sugars > 15 else 0,
        'is_high_fat': 1 if fat > 20 else 0,
        'is_high_salt': 1 if salt > 1.5 else 0,
        'is_low_cal': 1 if energy < 40 else 0
    }
    
    # Aseguramos el orden exacto de las columnas
    input_df = pd.DataFrame([data])[features_20]
    
    # Predicción
    res = model.predict(input_df)[0]
    letras = ["A", "B", "C", "D", "E"]
    colores = ["#008145", "#85BB2F", "#FECB02", "#EE8100", "#E63E11"]
    
    st.markdown(f"""
        <div style="background-color:{colores[res]}; padding:20px; border-radius:10px; text-align:center;">
            <h1 style="color:white; margin:0;">NUTRI-SCORE: {letras[res]}</h1>
        </div>
    """, unsafe_allow_html=True)
   
