# 🥗 Nutri-Score AI Predictor

[![Streamlit App](https://static.streamlit.io/badge_github_white.svg)](https://share.streamlit.io/) ## 📝 Descripción del Proyecto
Este proyecto presenta una aplicación web interactiva que automatiza la clasificación nutricional de alimentos utilizando **Inteligencia Artificial (Machine Learning)**. Basado en el sistema europeo *Nutri-Score*, el modelo analiza los componentes nutricionales por cada 100g de producto para predecir su categoría de calidad desde la **A** (Muy buena calidad) hasta la **E** (Baja calidad).

### Adaptación al Contexto Colombiano
A diferencia del sistema convencional europeo que requiere el dato estricto de sal en gramos, esta aplicación innova al permitir el ingreso directo de **Sodio en miligramos (mg)**, facilitando la lectura directa desde las tablas nutricionales reguladas en Colombia. El sistema realiza la conversión matemática en tiempo real para alimentar el algoritmo de predicción.

---

## 🚀 Características Clave
* **Algoritmo de Clasificación:** Desarrollado con **XGBoost Classifier**, optimizado mediante ingeniería de variables distribuidas en 20 características clave.
* **Rendimiento del Modelo:** El modelo alcanza un **Área Bajo la Curva (AUC) de 0.98**, demostrando una altísima robustez y confianza en la discriminación de categorías saludables frente a las nocivas.
* **IA Explicable (XAI):** Inspirado en las sugerencias de interpretabilidad de **SHAP Values (Shapley Additive exPlanations)**, la interfaz desglosa localmente qué nutrientes específicos actuaron como "atractores" hacia categorías saludables o "detractores" que penalizan la nota del alimento.
* **Diseño de Interfaz Customizado:** UI/UX moderna que adopta los colores institucionales (Verde Bosque y Salmón) integrando un contenedor de resultado dinámico y semáforo pedagógico.

---

## 📊 Estructura de Variables Analizadas (20 Features)
El modelo predice la clase (*Target: Nutri-Score*) evaluando los siguientes macronutrientes, micronutrientes y ratios calculados:

1.  `energy-kcal_100g` (Calorías)
2.  `fat_100g` (Grasas totales)
3.  `saturated-fat_100g` (Grasas saturadas)
4.  `carbohydrates_100g` (Carbohidratos totales)
5.  `sugars_100g` (Azúcares)
6.  `fiber_100g` (Fibra dietaria)
7.  `proteins_100g` (Proteínas)
8.  `salt_100g` (Sal calculada en gramos)
9.  `sodium_100g` (Sodio calculado en gramos)
10. `sugar_ratio` (Proporción de azúcares en carbohidratos)
11. `sat_fat_ratio` (Proporción de grasas saturadas)
12. `protein_ratio` (Densidad proteica)
13. `salt_density` (Densidad de sodio por caloría)
14. `carb_minus_sugar` (Carbohidratos complejos)
15. `total_solids` (Sólidos totales del producto)
16. `is_beverage` (Flag condicional para bebidas)
17. `is_high_sugar` (Alerta de azúcares altos)
18. `is_high_fat` (Alerta de grasas altas)
19. `is_high_salt` (Alerta de sodio crítico)
20. `is_low_cal` (Identificador de productos hipocalóricos)

---

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3
* **Librerías de ML/Data:** `xgboost`, `scikit-learn`, `pandas`, `joblib`
* **Despliegue e Interfaz:** `streamlit`
* **Diseño:** HTML5 / CSS3 embebido

---

## 📦 Instrucciones de Instalación y Uso Local

Si deseas clonar y ejecutar este predictor de manera local, sigue estos pasos:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
   cd TU_REPOSITORIO
