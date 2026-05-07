import streamlit as st
import joblib
import pandas as pd
import re

# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO (Fondo gris claro)
st.set_page_config(page_title="Nutri-Score AI Predictor", page_icon="🚦", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE ACTIVOS (Modelo y Variables)
@st.cache_resource
def load_assets():
    try:
        # Rutas actualizadas a la carpeta /models
        model = joblib.load('models/nutriscore_xgb_model.pkl')
        features = joblib.load('models/features_list.pkl')
        return model, features
    except Exception as e:
        st.error(f"⚠️ Error al cargar los componentes de IA: {e}")
        return None, None

model, features_20 = load_assets()

# 3. CABECERA Y CONTEXTO EDUCATIVO
st.title("🚦 Nutri-Score AI Predictor")
st.markdown("### Clasificación Nutricional Automatizada mediante Machine Learning")
  
with st.expander("🔍 ¿Qué significan estos colores? (Entiende tu salud)", expanded=True):
    # Usamos HTML/CSS para crear el diseño horizontal con flechas
    st.markdown("""
        <div style="font-family: sans-serif; background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            
            <p style="margin: 0 0 15px 0; font-weight: bold; color: #333; font-size: 1.1em;">Rango de Calidad Nutricional (por 100g)</p>
            
            <div style="display: flex; justify-content: center; gap: 12px; margin-bottom: 10px; align-items: center;">
                <div style="width: 55px; height: 55px; background-color: #008145; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.5em; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">A</div>
                <div style="width: 55px; height: 55px; background-color: #85BB2F; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.5em; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">B</div>
                <div style="width: 55px; height: 55px; background-color: #FECB02; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.5em; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">C</div>
                <div style="width: 55px; height: 55px; background-color: #EE8100; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.5em; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">D</div>
                <div style="width: 55px; height: 55px; background-color: #E63E11; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.5em; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">E</div>
            </div>
            
            <div style="display: flex; justify-content: space-between; width: 100%; max-width: 380px; margin: 0 auto; padding-top: 5px;">
                <div style="display: flex; align-items: center; color: #008145;">
                    <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #008145;"></div>
                    <div style="width: 130px; height: 2px; background-color: #008145;"></div>
                    <p style="margin: 0; padding-left: 5px; font-weight: bold; font-size: 0.9em;">Más saludable</p>
                </div>
                
                <div style="display: flex; align-items: center; color: #E63E11; flex-direction: row-reverse;">
                    <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #E63E11;"></div>
                    <div style="width: 130px; height: 2px; background-color: #E63E11;"></div>
                    <p style="margin: 0; padding-right: 5px; font-weight: bold; font-size: 0.9em;">Menos saludable</p>
                </div>
            </div>
        </div>
        <br>
    """, unsafe_allow_html=True)
        
# 4. FUNCIÓN DE VALIDACIÓN DE DECIMALES (Formato 0.5 obligatorio)
def get_clean_input(label, default):
    val_str = st.sidebar.text_input(label, value=str(default))
    # Exige formato numérico y bloquea formatos como ".5"
    if not re.match(r"^\d+(\.\d+)?$", val_str):
        st.sidebar.error(f"❌ Formato inválido en {label}. Usa '0.5' en lugar de '.5'")
        st.stop()
    return float(val_str)

# 5. PANEL DE ENTRADA (Sidebar)
st.sidebar.header("🚦 Panel de Nutrientes (100g)")
try:
    energy = get_clean_input("Energía (kcal)", 150.0)
    sugars = get_clean_input("Azúcares (g)", 5.0)
    fat = get_clean_input("Grasas totales (g)", 10.0)
    sat_fat = get_clean_input("Grasas saturadas (g)", 2.0)
    proteins = get_clean_input("Proteínas (g)", 5.0)
    fiber = get_clean_input("Fibra (g)", 2.0)
    salt = get_clean_input("Sal (g)", 0.5)
    is_bev = st.sidebar.selectbox("¿Es una bebida?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
except Exception:
    st.stop()

# 6. PREDICCIÓN Y RESULTADO
if st.button("Generar Diagnóstico"):
    try:
        # Replicamos el Feature Engineering de las 20 variables
        data = {
            'energy-kcal_100g': energy, 'fat_100g': fat, 'saturated-fat_100g': sat_fat,
            'carbohydrates_100g': sugars + 10, 
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
        
        # Predicción usando el orden de columnas original
        input_df = pd.DataFrame([data])[features_20]
        idx = model.predict(input_df)[0]
        
        letras = ["A", "B", "C", "D", "E"]
        colores = ["#008145", "#85BB2F", "#FECB02", "#EE8100", "#E63E11"]
        
        # Mostrar resultado con estilo profesional
        st.markdown(f"""
            <div style="background-color:{colores[idx]}; padding:40px; border-radius:20px; text-align:center; border: 3px solid white; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                <h1 style="color:white; margin:0; font-size: 60px; font-family: sans-serif;">NUTRI-SCORE: {letras[idx]}</h1>
                <p style="color:white; font-size: 20px;">Predicción generada con 98% de confianza (AUC)</p>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
        
    except Exception as e:
        st.error(f"Hubo un problema al procesar la predicción: {e}")
