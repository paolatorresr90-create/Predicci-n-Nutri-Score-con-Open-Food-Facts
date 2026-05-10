import streamlit as st
import joblib
import pandas as pd
import re
import os

# 1. CONFIGURACIÓN Y ESTILO GLOBAL (Sidebar Verde y Títulos Salmón)
st.set_page_config(page_title="Nutri-Score AI Predictor", page_icon="🥗")

st.markdown("""
    <style>
    .stApp { background-color: #fdfaf9; }
    
    /* SIDEBAR VERDE BOSQUE CON TEXTO CLARO */
    [data-testid="stSidebar"] { background-color: #2e5a42 !important; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { 
        color: #fdfaf9 !important; 
    }
    
    /* Títulos Salmón para la interfaz general */
    h1, h2 { color: #f5a191 !important; }

    /* Botón de Diagnóstico */
    .stButton>button {
        background-color: #f5a191;
        color: white;
        border-radius: 12px;
        height: 50px;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE MODELO (Con rutas flexibles para evitar errores)
@st.cache_resource
def load_assets():
    rutas_modelo = ['models/nutriscore_xgb_model.pkl', 'nutriscore_xgb_model.pkl']
    rutas_features = ['models/features_list.pkl', 'features_list.pkl']
    model, features = None, None
    for r in rutas_modelo:
        if os.path.exists(r): model = joblib.load(r); break
    for r in rutas_features:
        if os.path.exists(r): features = joblib.load(r); break
    return model, features

model, features_20 = load_assets()

# 3. SIDEBAR (Panel de Entrada)
st.sidebar.markdown('<p style="color:#f5a191; font-size:24px; font-weight:bold; text-align:center;">🥗 Panel de Nutrientes</p>', unsafe_allow_html=True)

def get_clean_input(label, default):
    val_str = st.sidebar.text_input(label, value=str(default))
    if not re.match(r"^\d+(\.\d+)?$", val_str):
        st.sidebar.error("❌ Usa punto para decimales")
        st.stop()
    return float(val_str)

energy = get_clean_input("Calorías (kcal)", 150.0)
sugars = get_clean_input("Azúcares (g)", 5.0)
fat = get_clean_input("Grasas totales (g)", 10.0)
sat_fat = get_clean_input("Grasas saturadas (g)", 2.0)
proteins = get_clean_input("Proteínas (g)", 5.0)
fiber = get_clean_input("Fibra (g)", 2.0)
sodium_mg = get_clean_input("Sodio (mg) - Etiqueta CO", 150.0)
is_bev = st.sidebar.selectbox("¿Es una bebida?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

# 4. CUERPO PRINCIPAL
st.title("🥗 Nutri-Score AI Predictor")

with st.expander("🔍 ¿Qué significan estos colores? (Entiende tu salud)", expanded=True):
    st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <div style="display: flex; justify-content: center; gap: 10px; margin-bottom: 15px;">
                <div style="width: 40px; height: 40px; background-color: #008145; border-radius: 5px; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;">A</div>
                <div style="width: 40px; height: 40px; background-color: #85BB2F; border-radius: 5px; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;">B</div>
                <div style="width: 40px; height: 40px; background-color: #FECB02; border-radius: 5px; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;">C</div>
                <div style="width: 40px; height: 40px; background-color: #EE8100; border-radius: 5px; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;">D</div>
                <div style="width: 40px; height: 40px; background-color: #E63E11; border-radius: 5px; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;">E</div>
            </div>
            <p style="color: #2e5a42; font-size: 0.9em; text-align: left; background-color: white; padding: 10px; border-radius: 10px; border-left: 5px solid #f5a191;">
                <strong>Nota del Modelo:</strong> Esta clasificación se basa en el análisis por 100g. 
                Es una herramienta comparativa para productos de la misma categoría.
            </p>
        </div>
    """, unsafe_allow_html=True)

# 5. PREDICCIÓN Y RESULTADO
if st.button("Generar Diagnóstico"):
    if model is not None:
        try:
            val_sal = sodium_mg / 400
            data = {
                'energy-kcal_100g': energy, 'fat_100g': fat, 'saturated-fat_100g': sat_fat,
                'carbohydrates_100g': sugars + 10, 'sugars_100g': sugars, 'fiber_100g': fiber,
                'proteins_100g': proteins, 'salt_100g': val_sal, 'sodium_100g': sodium_mg / 1000,
                'sugar_ratio': sugars / (sugars + 10.1), 'sat_fat_ratio': sat_fat / (fat + 0.1),
                'protein_ratio': proteins / (fat + proteins + 10.1), 'salt_density': val_sal / (energy + 1),
                'carb_minus_sugar': 10.0, 'total_solids': fat + sugars + proteins + val_sal,
                'is_beverage': is_bev, 'is_high_sugar': 1 if sugars > 15 else 0,
                'is_high_fat': 1 if fat > 20 else 0, 'is_high_salt': 1 if val_sal > 1.5 else 0,
                'is_low_cal': 1 if energy < 40 else 0
            }
            
            idx = model.predict(pd.DataFrame([data])[features_20])[0]
            letras = ["A", "B", "C", "D", "E"]
            colores = ["#008145", "#85BB2F", "#FECB02", "#EE8100", "#E63E11"]
            descripciones = [
                "Excelente calidad nutricional", "Buena calidad nutricional", 
                "Calidad nutricional aceptable", "Baja calidad nutricional", "Mala calidad nutricional"
            ]

# RESULTADO FINAL: Tarjeta con blanco forzado mediante selectores universales
            st.markdown(f"""
                <div style="background-color: {colores[idx]}; padding: 40px; border-radius: 25px; text-align: center; margin-top: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); color: white !important;">
                    <div style="margin-bottom: 10px;">
                        <span style="color: #FFFFFF !important; font-size: 110px; font-weight: bold; font-family: sans-serif; display: block; line-height: 1;">
                            {letras[idx]}
                        </span>
                    </div>
                    <div style="margin-top: 20px;">
                        <span style="color: #FFFFFF !important; font-size: 32px; font-weight: bold; font-family: sans-serif; display: block;">
                            {descripciones[idx]}
                        </span>
                    </div>
                    <hr style="border: 1px solid rgba(255,255,255,0.4); margin: 25px 0;">
                    <div style="color: #FFFFFF !important; font-size: 18px; opacity: 0.9; font-family: sans-serif; display: block;">
                        Análisis de Inteligencia Artificial basado en Nutri-Score
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"Hubo un problema al procesar la predicción: {e}")
    else:
        st.error("Modelo no cargado. Verifica tus archivos .pkl")
