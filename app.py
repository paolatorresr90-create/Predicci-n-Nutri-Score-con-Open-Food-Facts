import streamlit as st
import joblib
import pandas as pd
import re
import os

# 1. ESTILO Y CONFIGURACIÓN (Sidebar Verde Bosque #2e5a42)
st.set_page_config(page_title="Nutri-Score AI Predictor", page_icon="🥗")

st.markdown("""
    <style>
    /* Fondo General */
    .stApp { background-color: #fdfaf9; }
    
    /* 1. SIDEBAR VERDE BOSQUE */
    [data-testid="stSidebar"] { 
        background-color: #2e5a42 !important; 
    }
    
    /* 2. TEXTO DEL SIDEBAR (Mantenemos el color de tu imagen anterior) */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span { 
        color: #fdfaf9 !important; /* Color crema/blanco roto de tu captura */
    }

    /* Título del Sidebar en Salmón */
    .sidebar-title { 
        color: #f5a191 !important; 
        font-size: 24px; 
        font-weight: bold; 
        text-align: center; 
        margin-bottom: 20px; 
    }

    /* Títulos de la página central en Salmón */
    h1, h2 { color: #f5a191 !important; }

    /* Estilo para la Nota Técnica */
    .nota-expander {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #f5a191;
        font-size: 14px;
        color: #4a4e69;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE ACTIVOS (Con rutas flexibles para evitar errores)
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

# 3. SIDEBAR
st.sidebar.markdown('<p class="sidebar-title">🥗 Panel de Nutrientes</p>', unsafe_allow_html=True)

def get_clean_input(label, default):
    val_str = st.sidebar.text_input(label, value=str(default))
    if not re.match(r"^\d+(\.\d+)?$", val_str):
        st.sidebar.error(f"❌ Usa punto para decimales")
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
            <div style="display: flex; justify-content: center; gap: 8px; margin-bottom: 10px;">
                <div style="width: 35px; height: 35px; background-color: #008145; border-radius: 5px; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;">A</div>
                <div style="width: 35px; height: 35px; background-color: #85BB2F; border-radius: 5px; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;">B</div>
                <div style="width: 35px; height: 35px; background-color: #FECB02; border-radius: 5px; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;">C</div>
                <div style="width: 35px; height: 35px; background-color: #EE8100; border-radius: 5px; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;">D</div>
                <div style="width: 35px; height: 35px; background-color: #E63E11; border-radius: 5px; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;">E</div>
            </div>
            <div class="nota-expander">
                <strong>💡 Nota técnica:</strong> Este modelo predice con base en el estándar histórico (98% AUC). 
                El algoritmo oficial (2024) ha introducido mejoras para aceites y productos integrales.
            </div>
        </div>
    """, unsafe_allow_html=True)

# 5. LÓGICA DE PREDICCIÓN
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
            descripciones = ["Muy buena calidad nutricional", "Calidad nutricional buena", "Calidad nutricional media", "Baja calidad nutricional", "Mala calidad nutricional"]

            # RESULTADO: Cuadro sólido con letra blanca y texto del mismo color que el cuadro
            st.markdown(f"""
                <div style="display: flex; flex-direction: column; align-items: center; margin-top: 30px;">
                    <div style="background-color: {colores[idx]}; width: 120px; height: 120px; border-radius: 15px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <h1 style="color: white !important; font-size: 75px; margin: 0; font-family: sans-serif;">{letras[idx]}</h1>
                    </div>
                    <h2 style="color: {colores[idx]} !important; margin-top: 20px; text-align: center; font-family: sans-serif;">{descripciones[idx]}</h2>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        except Exception as e:
            st.error(f"Error: {e}")
       
