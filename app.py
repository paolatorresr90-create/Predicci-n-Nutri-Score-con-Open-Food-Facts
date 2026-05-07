import streamlit as st
import joblib
import pandas as pd
import re

# 1. ESTILO Y CONFIGURACIÓN (Fondo gris claro para resaltar los colores)
st.set_page_config(page_title="Nutri-Score AI Predictor", page_icon="🥗")
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE ACTIVOS CON MANEJO DE ERRORES
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/nutriscore_xgb_model.pkl')
        features = joblib.load('models/features_list.pkl')
        return model, features
    except Exception as e:
        st.error(f"Error al cargar el cerebro de la IA: {e}")
        return None, None

model, features_20 = load_assets()

# 3. CONTEXTO EDUCATIVO
st.title("🥗 Nutri-Score AI Predictor")
with st.expander("ℹ️ ¿Qué es el Nutri-Score?"):
    st.write("""
    El **Nutri-Score** es un sistema de clasificación de 5 colores (A a E) que mide la calidad nutricional 
    por cada 100g. La **A (verde)** es la opción más saludable, mientras que la **E (rojo)** indica 
    baja calidad nutricional por exceso de azúcares, grasas saturadas o sal.
    """)

# 4. FUNCIÓN DE VALIDACIÓN ESTRICTA
def get_valid_input(label, default):
    val_str = st.sidebar.text_input(label, value=str(default))
    # Regex para obligar formato 0.5 y bloquear .5
    if not re.match(r"^\d+(\.\d+)?$", val_str):
        st.sidebar.error(f"Error en {label}: Usa formato '0.5' (no '.5')")
        st.stop()
    return float(val_str)

# 5. FORMULARIO DE ENTRADA (Manejo de errores y tipos)
st.sidebar.header("Panel de Nutrientes (100g)")
try:
    energy = get_valid_input("Energía (kcal)", 150.0)
    sugars = get_valid_input("Azúcares (g)", 5.0)
    fat = get_valid_input("Grasas totales (g)", 10.0)
    sat_fat = get_valid_input("Grasas saturadas (g)", 2.0)
    proteins = get_valid_input("Proteínas (g)", 5.0)
    fiber = get_valid_input("Fibra (g)", 2.0)
    salt = get_valid_input("Sal (g)", 0.5)
    is_bev = st.sidebar.selectbox("¿Es una bebida?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
except Exception as e:
    st.sidebar.error("Error en la entrada de datos.")
    st.stop()

# 6. PREDICCIÓN CON LAS 20 VARIABLES
if st.button("Generar Diagnóstico"):
    try:
        # Replicamos el Feature Engineering exacto del entrenamiento
        data = {
            'energy-kcal_100g': energy, 'fat_100g': fat, 'saturated-fat_100g': sat_fat,
            'carbohydrates_100g': sugars + 10, # Estimación lógica
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
        
        input_df = pd.DataFrame([data])[features_20]
        res = model.predict(input_df)[0]
        
        letras = ["A", "B", "C", "D", "E"]
        colores = ["#008145", "#85BB2F", "#FECB02", "#EE8100", "#E63E11"]
        
        st.markdown(f"""
            <div style="background-color:{colores[res]}; padding:30px; border-radius:15px; text-align:center; border: 2px solid white;">
                <h1 style="color:white; margin:0; font-family:sans-serif;">NUTRI-SCORE: {letras[res]}</h1>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
        
    except Exception as e:
        st.error(f"Error en la predicción técnica: {e}")
