import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Nutri-Score Predictor AI", layout="wide")

# 2. CARGA DEL MODELO Y VARIABLES
@st.cache_resource 
def load_assets():
    # Asegúrate de que las rutas coincidan con tu estructura de carpetas
    model = joblib.load('models/nutriscore_xgb_model.pkl')
    features = joblib.load('models/features_list.pkl')
    return model, features

model, features_20 = load_assets()

# 3. INTERFAZ DE USUARIO
st.title("🥗 Predictor Nutri-Score con IA")
st.markdown("""
Esta aplicación utiliza un modelo **XGBoost** entrenado con más de 80,000 registros para predecir 
la calidad nutricional de un producto basándose en su composición química.
""")

st.sidebar.header("Ingresa los Valores Nutricionales (por 100g)")

with st.sidebar:
    energy = st.number_input("Energía (kcal)", 0.0, 900.0, 150.0)
    fat = st.number_input("Grasas totales (g)", 0.0, 100.0, 10.0)
    sat_fat = st.number_input("Grasas saturadas (g)", 0.0, 100.0, 2.0)
    sugars = st.number_input("Azúcares (g)", 0.0, 100.0, 5.0)
    carbs = st.number_input("Carbohidratos (g)", 0.0, 100.0, 20.0)
    fiber = st.number_input("Fibra (g)", 0.0, 50.0, 2.0)
    proteins = st.number_input("Proteínas (g)", 0.0, 100.0, 5.0)
    salt = st.number_input("Sal (g)", 0.0, 40.0, 0.5)
    is_bev = st.checkbox("¿Es una bebida?")

# 4. LÓGICA DE FEATURE ENGINEERING (Idéntica al entrenamiento)
def predict_score():
    data = {
        'energy-kcal_100g': energy, 'fat_100g': fat, 'saturated-fat_100g': sat_fat,
        'carbohydrates_100g': carbs, 'sugars_100g': sugars, 'fiber_100g': fiber,
        'proteins_100g': proteins, 'salt_100g': salt, 'sodium_100g': salt/2.5,
        'sugar_ratio': sugars/(carbs+0.1), 'sat_fat_ratio': sat_fat/(fat+0.1),
        'protein_ratio': proteins/(fat+carbs+proteins+0.1), 'salt_density': salt/(energy+1),
        'carb_minus_sugar': carbs - sugars, 'total_solids': fat+carbs+proteins+salt,
        'is_beverage': 1 if is_bev else 0, 'is_high_sugar': 1 if sugars > 15 else 0,
        'is_high_fat': 1 if fat > 20 else 0, 'is_high_salt': 1 if salt > 1.5 else 0,
        'is_low_cal': 1 if energy < 40 else 0
    }
    
    input_df = pd.DataFrame([data])[features_20]
    prediction = model.predict(input_df)[0]
    
    letras = ['A', 'B', 'C', 'D', 'E']
    colores = ['#008145', '#85BB2F', '#FECB02', '#EE8100', '#E63E11']
    
    return letras[prediction], colores[prediction]

# 5. BOTÓN DE ACCIÓN Y RESULTADO
if st.button("Calcular Nutri-Score"):
    letra, color = predict_score()
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color}; color: white;">
        <h1 style="margin: 0; font-size: 50px;">GRADO {letra}</h1>
    </div>
    """, unsafe_allow_html=True)
    st.balloons()