"""
streamlit_app.py  –  App de predicción de contratación de depósito bancario
=========================================================================
Práctica 1 · Aprendizaje Automático · UC3M · 2025-26
Autores: Marcos Truchado (100522249) · Jorge Condado (100522327)

Uso:
    python -m streamlit run app/streamlit_app.py

Requisitos: (IMPORTANTES)
    - modelo_final.joblib en la carpeta models/ del repositorio.
    - Misma versión de scikit-learn que al guardar el modelo (p. ej. 1.8.x).
    - Es muy recomendable usar el repositorio entero desde github, en caso de que 
    - algo no funcione es debido a que no esta bien introducidos los archivos, 
    - se recomienda coger el repositorio entero desde github y descargar los archivos necesarios.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


# Transformadores custom del pipeline (mismos nombres/código que en Notebook 1/2);
# hacen falta para deserializar el .joblib generado en Jupyter.

class PdaysTransformer(BaseEstimator, TransformerMixin):
    """Crea contacted_before y sustituye pdays=-1 por 0 (idempotente)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        needs_transform = 'contacted_before' not in X.columns or (X['pdays'] == -1).any()
        if not needs_transform:
            return X
        X = X.copy()
        X['contacted_before'] = (X['pdays'] != -1).astype(int)
        X['pdays'] = X['pdays'].replace(-1, 0)
        return X


class MaritalImputer(BaseEstimator, TransformerMixin):
    """Imputa NaN de marital con la moda aprendida en fit (idempotente)."""
    def fit(self, X, y=None):
        self.mode_ = X['marital'].mode()[0]
        return self
    def transform(self, X):
        if not X['marital'].isna().any():
            return X
        X = X.copy()
        X['marital'] = X['marital'].fillna(self.mode_)
        return X

# ==============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ==============================================================================

# Ruta al modelo guardado. Ajustar si está en otra ubicación.
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "modelo_final.joblib"

# Categorías posibles de cada variable categórica
# (extraídas del EDA del Notebook 1)
JOBS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"
]
MARITAL = ["divorced", "married", "single"]
EDUCATION = ["primary", "secondary", "tertiary", "unknown"]
CONTACT = ["cellular", "telephone", "unknown"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME = ["failure", "other", "success", "unknown"]
YES_NO = ["no", "yes"]

INSTANCIA_DEMO_1 = {
    "label": "Instancia 1 (age=36, technician, duration=441s — esperado: NO)",
    "data": {
        "age": 36, "job": "technician", "marital": "married",
        "education": "secondary", "default": "no", "balance": 1466,
        "housing": "yes", "loan": "no", "contact": "unknown",
        "day": 29, "month": "may", "duration": 441, "campaign": 1,
        "pdays": -1, "previous": 0, "poutcome": "unknown",
    }
}
INSTANCIA_DEMO_2 = {
    "label": "Instancia 2 (age=41, entrepreneur, duration=438s — esperado: SI)",
    "data": {
        "age": 41, "job": "entrepreneur", "marital": "married",
        "education": "primary", "default": "no", "balance": 0,
        "housing": "no", "loan": "no", "contact": "cellular",
        "day": 3, "month": "jul", "duration": 438, "campaign": 2,
        "pdays": 389, "previous": 1, "poutcome": "other",
    }
}


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

@st.cache_resource
def cargar_modelo():
    """Carga el pipeline guardado con joblib."""
    return joblib.load(MODEL_PATH)


def construir_dataframe(inputs: dict) -> pd.DataFrame:
    """
    Recibe el diccionario de entradas del formulario y devuelve un DataFrame
    crudo de una fila. El pipeline del modelo se encarga internamente de
    transformar pdays y de imputar marital (via PdaysTransformer y MaritalImputer).
    """
    fila = {
        "age":      inputs["age"],
        "job":      inputs["job"],
        "marital":  inputs["marital"],
        "education":inputs["education"],
        "default":  inputs["default"],
        "balance":  inputs["balance"],
        "housing":  inputs["housing"],
        "loan":     inputs["loan"],
        "contact":  inputs["contact"],
        "day":      inputs["day"],
        "month":    inputs["month"],
        "duration": inputs["duration"],
        "campaign": inputs["campaign"],
        "pdays":    inputs["pdays"],
        "previous": inputs["previous"],
        "poutcome": inputs["poutcome"],
    }
    return pd.DataFrame([fila])


def predecir(modelo, df_input: pd.DataFrame) -> tuple[str, float]:
    """
    Aplica el modelo a un DataFrame de una fila.

    Returns
    -------
    (etiqueta, probabilidad_de_sí)
        etiqueta   : 'SÍ contratará' o 'NO contratará'
        probabilidad: float en [0, 1]
    """
    pred_num = modelo.predict(df_input)[0]       # 0 o 1
    prob = modelo.predict_proba(df_input)[0][1]  # prob. de clase 1 (sí)
    etiqueta = "SÍ contratará el depósito" if pred_num == 1 else "❌  NO contratará el depósito"
    return etiqueta, prob


# ==============================================================================
# INTERFAZ DE USUARIO
# ==============================================================================

def main():
    # ── Configuración de página ──────────────────────────────────────────────
    st.set_page_config(
        page_title="Predictor de Depósito Bancario",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 Predictor de Subscripción a Depósito Bancario")
    st.markdown(
        "Esta aplicación utiliza un modelo **SVM con kernel RBF** entrenado "
        "sobre datos de campañas de marketing bancario para predecir si un "
        "cliente contratará un depósito a plazo."
    )
    st.markdown("---")

    # ── Carga del modelo ─────────────────────────────────────────────────────
    try:
        modelo = cargar_modelo()
    except FileNotFoundError:
        st.error(
            f"No se encontró el archivo del modelo en '{MODEL_PATH}'. "
            "Asegúrate de que 'models/modelo_final.joblib' está disponible "
            "en la raíz del repositorio."
        )
        st.stop()

    # ── Pestañas principales ─────────────────────────────────────────────────
    tab_pred, tab_demo = st.tabs(
        ["🔍 Predicción para nuevo cliente", "Verificación del modelo"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: FORMULARIO DE PREDICCIÓN
    # ══════════════════════════════════════════════════════════════════════════
    with tab_pred:
        st.subheader("Introduce los datos del cliente")
        st.caption(
            "Rellena todos los campos y pulsa **Predecir** para obtener la "
            "estimación del modelo."
        )

        # ── Sección 1: Datos demográficos ────────────────────────────────────
        st.markdown("##### 👤 Datos personales")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Edad (años)", min_value=18, max_value=100, value=40, step=1,
                help="Edad del cliente en años."
            )
            job = st.selectbox(
                "Tipo de trabajo", options=JOBS, index=JOBS.index("admin."),
                help="Categoría laboral del cliente."
            )
        with col2:
            marital = st.selectbox(
                "Estado civil", options=MARITAL, index=MARITAL.index("married")
            )
            education = st.selectbox(
                "Nivel de educación", options=EDUCATION,
                index=EDUCATION.index("secondary")
            )
        with col3:
            default = st.selectbox(
                "¿Tiene crédito impagado?", options=YES_NO,
                index=YES_NO.index("no"),
                help="Indica si el cliente tiene algún crédito no devuelto."
            )
            balance = st.number_input(
                "Balance anual medio (€)", min_value=-10_000, max_value=200_000,
                value=1_000, step=100,
                help="Saldo bancario anual medio del cliente."
            )

        st.markdown("##### 🏠 Situación financiera")
        col4, col5 = st.columns(2)
        with col4:
            housing = st.selectbox(
                "¿Tiene hipoteca?", options=YES_NO, index=YES_NO.index("no")
            )
        with col5:
            loan = st.selectbox(
                "¿Tiene préstamo personal?", options=YES_NO,
                index=YES_NO.index("no")
            )

        # ── Sección 2: Último contacto de la campaña ─────────────────────────
        st.markdown("##### 📞 Datos del último contacto (campaña actual)")
        col6, col7, col8 = st.columns(3)

        with col6:
            contact = st.selectbox(
                "Tipo de contacto", options=CONTACT,
                index=CONTACT.index("cellular"),
                help="Medio de comunicación utilizado."
            )
            month = st.selectbox(
                "Mes del último contacto", options=MONTHS,
                index=MONTHS.index("may"),
                help="Mes del año en que se realizó el último contacto."
            )
        with col7:
            day = st.number_input(
                "Día del mes del último contacto", min_value=1, max_value=31,
                value=15, step=1
            )
            duration = st.number_input(
                "Duración del último contacto (segundos)",
                min_value=0, max_value=5_000, value=300, step=10,
                help=(
                    "Duración en segundos de la última llamada. "
                    "Este dato no se conoce antes de realizar la llamada, "
                    "por lo que su uso es solo retrospectivo."
                )
            )
        with col8:
            campaign = st.number_input(
                "Nº de contactos en esta campaña", min_value=1, max_value=100,
                value=1, step=1,
                help="Número total de veces que se ha contactado al cliente "
                     "durante la campaña actual."
            )

        # ── Sección 3: Historial de campañas anteriores ───────────────────────
        st.markdown("##### Historial de campañas anteriores")
        col9, col10, col11 = st.columns(3)

        with col9:
            pdays = st.number_input(
                "pdays: dias desde contacto anterior",
                min_value=-1, max_value=1000, value=-1, step=1,
                help=(
                    "Numero de dias desde que el cliente fue contactado en la "
                    "campana anterior. Introduce **-1** si nunca fue contactado "
                    "o si se desconoce."
                )
            )
            if pdays == -1:
                st.info("pdays=-1 -> sin contacto previo")
            else:
                st.success(f"contactado hace {pdays} dias")

        with col10:
            previous = st.number_input(
                "Nº de contactos en campañas anteriores",
                min_value=0, max_value=100, value=0, step=1
            )
        with col11:
            poutcome = st.selectbox(
                "Resultado de campaña anterior", options=POUTCOME,
                index=POUTCOME.index("unknown"),
                help="Resultado de la última campaña de marketing con este cliente."
            )

        # ── Botón de predicción ───────────────────────────────────────────────
        st.markdown("---")
        if st.button("Predecir", type="primary", use_container_width=True):
            inputs = {
                "age": int(age), "job": job, "marital": marital,
                "education": education, "default": default,
                "balance": int(balance), "housing": housing, "loan": loan,
                "contact": contact, "day": int(day), "month": month,
                "duration": int(duration), "campaign": int(campaign),
                "pdays": int(pdays),
                "previous": int(previous), "poutcome": poutcome,
            }

            df_input = construir_dataframe(inputs)
            etiqueta, prob = predecir(modelo, df_input)

            # Resultado principal
            st.markdown("### Resultado de la predicción")
            if prob >= 0.5:
                st.success(etiqueta)
            else:
                st.error(etiqueta)

            # Barra de probabilidad
            st.metric(
                label="Probabilidad estimada de contratación",
                value=f"{prob * 100:.1f} %"
            )
            st.progress(float(prob))

            # DataFrame con los datos tal como los ve el modelo
            with st.expander("Ver datos enviados al modelo"):
                st.dataframe(df_input, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: VERIFICACIÓN (2 instancias del Notebook 2)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_demo:
        st.subheader("Verificación de consistencia con el Notebook 2")
        st.markdown(
            "El enunciado exige demostrar que la app produce **las mismas "
            "predicciones** que el modelo cargado directamente desde Python. "
            "A continuación se muestran las dos instancias de verificación "
            "definidas en la celda 7 del Notebook 2."
        )
        st.markdown("---")

        for i, demo in enumerate([INSTANCIA_DEMO_1, INSTANCIA_DEMO_2], start=1):
            st.markdown(f"#### {demo['label']}")

            datos = demo["data"]
            df_demo = construir_dataframe(datos)

            etiqueta, prob = predecir(modelo, df_demo)

            col_a, col_b = st.columns([1, 2])
            with col_a:
                if prob >= 0.5:
                    st.success(f"**Predicción app:** {etiqueta}")
                else:
                    st.error(f"**Predicción app:** {etiqueta}")
                st.metric(
                    "Probabilidad de contratación",
                    f"{prob * 100:.1f} %"
                )

            with col_b:
                st.caption("Datos crudos al pipeline (pdays/contacted_before los aplica el modelo):")
                st.dataframe(df_demo, use_container_width=True)

            # Resultado esperado según Notebook 2
            esperado = "yes" if i == 2 else "no"
            pred_raw = "yes" if prob >= 0.5 else "no"
            if pred_raw == esperado:
                st.success(
                    f"Coincide con el Notebook 2 (esperado: **{esperado}**)"
                )
            else:
                st.error(
                    f"¡NO coincide! App predice **{pred_raw}**, "
                    f"Notebook 2 predice **{esperado}**."
                )
            st.markdown("---")

       

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    main()
