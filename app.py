import streamlit as st
import joblib
import pandas as pd
import re

# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO 
# ESTILO GLOBAL (Sin variables dinámicas para evitar el NameError)
st.markdown("""
    <style>
    .stApp { background-color: #fdfaf9; }
    [data-testid="stSidebar"] { 
        background-color: #fdfaf9; 
        border-right: 2px solid #2e5a42; 
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
        color: #2e5a42 !important;
        font-weight: 600 !important;
    }
    h1, h2, .sidebar-title { color: #f5a191 !important; }
    
    /* Estilo para la Nota del Modelo */
    .nota-expander {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #f5a191;
        font-size: 14px;
        color: #4a4e69;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# EL EXPANDER CON LA NOTA
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
            <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #2e5a42; font-weight: bold;">
                <span>← Más saludable</span>
                <span>Menos saludable →</span>
            </div>
            <div class="nota-expander">
                <strong>💡 Nota del Modelo:</strong> Esta clasificación se basa en el análisis de nutrientes por 100g. 
                Recuerda que el Nutri-Score es una herramienta para comparar productos de la misma categoría.
            </div>
        </div>
    """, unsafe_allow_html=True)

# 3. BLOQUE DE RESULTADO (Cuadro sólido pequeño con letra blanca)
# Dentro del bloque de predicción (cuando ya tienes 'idx')
letras = ["A", "B", "C", "D", "E"]
colores = ["#008145", "#85BB2F", "#FECB02", "#EE8100", "#E63E11"]
descripciones = ["Muy buena calidad", "Buena calidad", "Calidad media", "Baja calidad", "Mala calidad"]

st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center; margin-top: 20px;">
        <div style="background-color: {colores[idx]}; width: 100px; height: 100px; border-radius: 15px; display: flex; align-items: center; justify-content: center;">
            <h1 style="color: white !important; font-size: 65px; margin: 0;">{letras[idx]}</h1>
        </div>
        <h2 style="color: #2e5a42 !important; margin-top: 15px;">{descripciones[idx]}</h2>
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center; margin-top: 20px;">
        <div style="
            background-color: {colores[idx]}; 
            width: 100px; 
            height: 100px; 
            border-radius: 15px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h1 style="color: white !important; font-size: 65px; margin: 0; font-family: sans-serif;">{letras[idx]}</h1>
        </div>
        <h2 style="color: #2e5a42 !important; margin-top: 15px; text-align: center;">{descripciones[idx]}</h2>
    </div>
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
st.title("🥗 Nutri-Score AI Predictor")
st.markdown("### Clasificación Nutricional Automatizada mediante Machine Learning")
  
with st.expander("🔍 ¿Qué significan estos colores? (Entiende tu salud)", expanded=True):
    st.markdown("""<div style="font-family: sans-serif; background-color: #ffffff; padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #e0e0e0;">
            <p style="margin: 0 0 15px 0; font-weight: bold; color: #444; font-size: 1.1em;">Rango de Calidad Nutricional (por 100g)</p>
            <div style="display: flex; justify-content: center; gap: 12px; margin-bottom: 15px;">
                <div style="width: 50px; height: 50px; background-color: #008145; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.3em;">A</div>
                <div style="width: 50px; height: 50px; background-color: #85BB2F; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.3em;">B</div>
                <div style="width: 50px; height: 50px; background-color: #FECB02; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.3em;">C</div>
                <div style="width: 50px; height: 50px; background-color: #EE8100; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.3em;">D</div>
                <div style="width: 50px; height: 50px; background-color: #E63E11; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.3em;">E</div>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0 10px;">
                <span style="color: #008145; font-weight: bold; font-size: 0.85em;">← Más saludable</span>
                <span style="color: #E63E11; font-weight: bold; font-size: 0.85em;">Menos saludable →</span>
            </div>
        </div>""", unsafe_allow_html=True)
        
# 4. FUNCIÓN DE VALIDACIÓN DE DECIMALES (Formato 0.5 obligatorio)
def get_clean_input(label, default):
    val_str = st.sidebar.text_input(label, value=str(default))
    # Exige formato numérico y bloquea formatos como ".5"
    if not re.match(r"^\d+(\.\d+)?$", val_str):
        st.sidebar.error(f"❌ Formato inválido en {label}. Usa '0.5' en lugar de '.5'")
        st.stop()
    return float(val_str)

# 5. PANEL DE ENTRADA (Sidebar)
st.sidebar.markdown('<p class="sidebar-title">🥗 Panel de Nutrientes</p>', unsafe_allow_html=True)
try:
    energy = get_clean_input("Calorías (kcal)", 150.0)
    sugars = get_clean_input("Azúcares (g)", 5.0)
    fat = get_clean_input("Grasas totales (g)", 10.0)
    sat_fat = get_clean_input("Grasas saturadas (g)", 2.0)
    proteins = get_clean_input("Proteínas (g)", 5.0)
    fiber = get_clean_input("Fibra (g)", 2.0)
    
    # Aquí es donde integramos el Sodio (mg) para Colombia
    sodium_mg = get_clean_input("Sodio (mg) - Según etiqueta CO", 150.0)
    
    is_bev = st.sidebar.selectbox("¿Es una bebida?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

except Exception:
    # Este bloque ya maneja cualquier error de las líneas de arriba
    st.stop()

# 6. PREDICCIÓN Y RESULTADO
if st.button("Generar Diagnóstico"):
    try:
        # 1. Convertimos el Sodio (mg) ingresado a Sal (g) para el modelo
        # Fórmula: Sal = Sodio (mg) / 400
        valor_sal_calculado = sodium_mg / 400
        valor_sodio_g = sodium_mg / 1000

        # 2. Replicamos las 20 variables usando el nuevo valor
        data = {
            'energy-kcal_100g': energy, 
            'fat_100g': fat, 
            'saturated-fat_100g': sat_fat,
            'carbohydrates_100g': sugars + 10, 
            'sugars_100g': sugars, 
            'fiber_100g': fiber, 
            'proteins_100g': proteins, 
            'salt_100g': valor_sal_calculado,   # <-- Aquí ya no usamos 'salt'
            'sodium_100g': valor_sodio_g,       # <-- Usamos el valor convertido
            'sugar_ratio': sugars / (sugars + 10.1),
            'sat_fat_ratio': sat_fat / (fat + 0.1),
            'protein_ratio': proteins / (fat + proteins + 10.1),
            'salt_density': valor_sal_calculado / (energy + 1),
            'carb_minus_sugar': 10.0,
            'total_solids': fat + sugars + proteins + valor_sal_calculado,
            'is_beverage': is_bev,
            'is_high_sugar': 1 if sugars > 15 else 0,
            'is_high_fat': 1 if fat > 20 else 0,
            'is_high_salt': 1 if valor_sal_calculado > 1.5 else 0,
            'is_low_cal': 1 if energy < 40 else 0
        }
                
        # Predicción usando el orden de columnas original
        input_df = pd.DataFrame([data])[features_20]
        idx = model.predict(input_df)[0]
        
    # 1. Definimos los datos del resultado
        letras = ["A", "B", "C", "D", "E"]
        colores = ["#008145", "#85BB2F", "#FECB02", "#EE8100", "#E63E11"]
        descripciones = [
            "Muy buena calidad nutricional",
            "Calidad nutricional buena",
            "Calidad nutricional media",
            "Baja calidad nutricional",
            "Mala calidad nutricional"
        ]
        
        # 2. Bloque de Resultado con estilo "Glassmorphism" (Efecto Cristal)
        st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.7); 
                backdrop-filter: blur(10px); 
                border-radius: 25px; 
                padding: 40px; 
                border: 4px solid {colores[idx]}; 
                text-align: center; 
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
                margin-bottom: 20px;">
                <h1 style="color:{colores[idx]}; font-size: 80px; margin: 0; font-family: sans-serif;">{letras[idx]}</h1>
                <h2 style="color: #4a4e69; margin: 10px 0; font-family: sans-serif;">{descripciones[idx]}</h2>
                <p style="color: #6c757d; font-size: 14px;">Predicción optimizada para etiquetado colombiano</p>
            </div>
            
            <div style="background-color: #fdfaf9; padding: 15px; border-radius: 10px; border-left: 5px solid {colores[idx]}; font-size: 14px; color: #4a4e69; font-family: sans-serif;">
                <strong>💡 Nota del Modelo:</strong> Esta clasificación se basa en el análisis de nutrientes por 100g. 
                Recuerda que el Nutri-Score es una herramienta complementaria para comparar productos de la misma categoría.
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
        
    except Exception as e:
        st.error(f"Hubo un problema al procesar la predicción: {e}")
