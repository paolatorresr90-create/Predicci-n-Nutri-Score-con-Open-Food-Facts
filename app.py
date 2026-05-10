import streamlit as st
import joblib
import pandas as pd
import re

# 1. ESTILO GLOBAL Y CONFIGURACIÓN (Colores de tu presentación Genially)
st.set_page_config(page_title="Nutri-Score AI Predictor", page_icon="🥗")

st.markdown("""
    <style>
    /* Fondo General */
    .stApp { background-color: #fdfaf9; }
    
    /* Sidebar: Fondo crema claro y borde verde bosque */
    [data-testid="stSidebar"] { 
        background-color: #fdfaf9; 
        border-right: 2px solid #2e5a42; 
    }

    /* Textos del Sidebar en Verde Bosque para legibilidad */
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
        color: #2e5a42 !important;
        font-weight: 600 !important;
    }

    /* Título del Panel de Nutrientes en Salmón */
    .sidebar-title { 
        color: #f5a191 !important; 
        font-size: 24px; 
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        border-bottom: 2px solid #f5a191;
    }

    /* Títulos Principales en Salmón */
    h1, h2 { color: #f5a191 !important; }
    
    /* Estilo para la Nota Técnica dentro del Expander */
    .nota-expander {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #f5a191;
        font-size: 14px;
        color: #4a4e69;
        margin-top: 10px;
    }
    
    /* Botón de Diagnóstico Estilo Pro */
    .stButton>button {
        background-color: #2e5a42;
        color: white;
        border-radius: 12px;
        width: 100%;
        height: 50px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE ACTIVOS DE IA
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/nutriscore_xgb_model.pkl')
        features = joblib.load('models/features_list.pkl')
        return model, features
    except Exception as e:
        st.error(f"⚠️ Error al cargar los componentes de IA: {e}")
        return None, None

model, features_20 = load_assets()

# 3. INTERFAZ PRINCIPAL (Parte Superior)
st.title("🥗 Nutri-Score AI Predictor")
st.markdown("### Clasificación Nutricional Automatizada mediante Machine Learning")

# Menú desplegable con la Nota del Modelo integrada
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
            <div style="display: flex; justify-content: space-between; font-size: 0.85em; color: #2e5a42; font-weight: bold; padding: 0 10px;">
                <span>← Más saludable</span>
                <span>Menos saludable →</span>
            </div>
            <div class="nota-expander">
                <strong>💡 Nota del Modelo:</strong> Esta clasificación se basa en el análisis de nutrientes por 100g. 
                Recuerda que el Nutri-Score es una herramienta para comparar productos de la misma categoría.
            </div>
        </div>
    """, unsafe_allow_html=True)

# 4. FUNCIONES Y SIDEBAR
def get_clean_input(label, default):
    val_str = st.sidebar.text_input(label, value=str(default))
    if not re.match(r"^\d+(\.\d+)?$", val_str):
        st.sidebar.error(f"❌ Usa '.' para decimales (ej. 0.5)")
        st.stop()
    return float(val_str)

st.sidebar.markdown('<p class="sidebar-title">🥗 Panel de Nutrientes</p>', unsafe_allow_html=True)

try:
    energy = get_clean_input("Calorías (kcal)", 150.0)
    sugars = get_clean_input("Azúcares (g)", 5.0)
    fat = get_clean_input("Grasas totales (g)", 10.0)
    sat_fat = get_clean_input("Grasas saturadas (g)", 2.0)
    proteins = get_clean_input("Proteínas (g)", 5.0)
    fiber = get_clean_input("Fibra (g)", 2.0)
    sodium_mg = get_clean_input("Sodio (mg) - Etiqueta CO", 150.0)
    is_bev = st.sidebar.selectbox("¿Es una bebida?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
except Exception:
    st.stop()

# 5. LÓGICA DE PREDICCIÓN
if st.button("Generar Diagnóstico"):
    if model is not None:
        try:
            # Conversión de Sodio (mg) a Sal (g)
            valor_sal_g = sodium_mg / 400
            valor_sodio_g = sodium_mg / 1000

            data = {
                'energy-kcal_100g': energy, 'fat_100g': fat, 'saturated-fat_100g': sat_fat,
                'carbohydrates_100g': sugars + 10, 'sugars_100g': sugars, 'fiber_100g': fiber,
                'proteins_100g': proteins, 'salt_100g': valor_sal_g, 'sodium_100g': valor_sodio_g,
                'sugar_ratio': sugars / (sugars + 10.1), 'sat_fat_ratio': sat_fat / (fat + 0.1),
                'protein_ratio': proteins / (fat + proteins + 10.1), 'salt_density': valor_sal_g / (energy + 1),
                'carb_minus_sugar': 10.0, 'total_solids': fat + sugars + proteins + valor_sal_g,
                'is_beverage': is_bev, 'is_high_sugar': 1 if sugars > 15 else 0,
                'is_high_fat': 1 if fat > 20 else 0, 'is_high_salt': 1 if valor_sal_g > 1.5 else 0,
                'is_low_cal': 1 if energy < 40 else 0
            }
            
            input_df = pd.DataFrame([data])[features_20]
            idx = model.predict(input_df)[0]
            
            letras = ["A", "B", "C", "D", "E"]
            colores = ["#008145", "#85BB2F", "#FECB02", "#EE8100", "#E63E11"]
            descripciones = [
                "Muy buena calidad nutricional", "Calidad nutricional buena",
                "Calidad nutricional media", "Baja calidad nutricional", "Mala calidad nutricional"
            ]

            # RESULTADO FINAL: Cuadro sólido con letra blanca
            st.markdown(f"""
                <div style="display: flex; flex-direction: column; align-items: center; margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.5); border-radius: 20px;">
                    <div style="
                        background-color: {colores[idx]}; 
                        width: 120px; 
                        height: 120px; 
                        border-radius: 20px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        box-shadow: 0 10px 20px rgba(0,0,0,0.15);">
                        <h1 style="color: white !important; font-size: 80px; margin: 0; font-family: sans-serif;">{letras[idx]}</h1>
                    </div>
                    <h2 style="color: #2e5a42 !important; margin-top: 20px; text-align: center;">{descripciones[idx]}</h2>
                    <p style="color: #6c757d; font-size: 14px;">Diagnóstico basado en IA (XGBoost)</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()

        except Exception as e:
            st.error(f"Error en la predicción: {e}")
    else:
        st.error("El modelo no está cargado. Revisa la carpeta /models.")
       
