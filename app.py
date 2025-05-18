import streamlit as st
import pandas as pd
import joblib
import sqlite3
from database import init_db, log_prediction, fetch_logs

# Intentamos importar SHAP y Matplotlib, si fallan desactivamos esa sección
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- Configuración de la página ---
st.set_page_config(
    page_title="AIGNITE – Evaluador de Riesgo de Incendio",
    page_icon="🔥",
    layout="wide"
)

# --- Inicializar base de datos y modelo ---
init_db()

@st.cache_resource
def load_model():
    bundle = joblib.load("models/aignite_model.pkl")
    return bundle['model'], bundle['columns'], bundle['cat_cols']

clf, model_columns, cat_cols = load_model()

if SHAP_AVAILABLE:
    @st.cache_resource
    def load_explainer():
        return shap.TreeExplainer(clf)
    explainer = load_explainer()

# --- Funciones ---
def predict(data: dict):
    df = pd.DataFrame([data])
    for c in cat_cols:
        df[c] = df[c].astype(str)
    df_enc = pd.get_dummies(df, columns=cat_cols).reindex(
        columns=model_columns, fill_value=0
    )
    pred = clf.predict(df_enc)[0]
    proba = clf.predict_proba(df_enc)[0]
    return pred, proba, df_enc

# Colores y iconos
RISK_STYLE = {
    'Bajo':  {'func': st.success, 'icon': '🟢'},
    'Medio': {'func': st.warning, 'icon': '🟡'},
    'Alto':  {'func': st.error,   'icon': '🔴'}
}

# --- Navegación ---
page = st.sidebar.radio("🏠 Navegación", [
    "Evaluar",
    "CRUD",
    "Histórico",
    "Explicabilidad",
    "Ayuda",
    "Dashboard",
    "Retrain"
])

# --- Página: Evaluar ---
if page == "Evaluar":
    st.title("🔥 AIGNITE – Evaluar Riesgo")
    with st.sidebar.expander("Parámetros", expanded=True):
        inputs = {
            'HEAT_SOURC': st.selectbox("Fuente de calor",   ['70','8','22','31','52']),
            'TYPE_MAT'  : st.selectbox("Material combustible", ['Madera','Hormigón','Metal','Plástico']),
            'STRUC_STAT': st.selectbox("Estado estructural", ['Bueno','Regular','Malo']),
            'DETECTOR'  : st.radio("Detector presente",     ['Y','N']),
            'DET_TYPE'  : st.selectbox("Tipo de detector",   ['1','2','3']),
            'AREA'      : st.slider("Superficie (m²)",      1, 10000, 100)
        }
        if st.button("Evaluar riesgo"):
            risk, proba, _ = predict(inputs)
            log_prediction(inputs, risk, proba)
            style = RISK_STYLE.get(risk)
            style['func'](f"{style['icon']} Nivel de riesgo: **{risk}**")
            st.write(f"🟢 Bajo: {proba[0]:.1%} | 🟡 Medio: {proba[1]:.1%} | 🔴 Alto: {proba[2]:.1%}")
            if risk == 'Bajo':
                st.balloons()

# --- Página: CRUD ---
elif page == "CRUD":
    st.title("🔧 CRUD de Registros")
    rec_id = st.number_input("ID del registro", min_value=1, step=1)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Consultar"):
            rec = next((r for r in fetch_logs() if r['id']==rec_id), None)
            if rec:
                st.json(rec)
            else:
                st.warning("Registro no encontrado.")
    with col2:
        if st.button("Eliminar"):
            conn=sqlite3.connect("incendios.db"); c=conn.cursor()
            c.execute("DELETE FROM registros_incendios WHERE id=?", (rec_id,))
            conn.commit(); conn.close()
            st.info(f"Registro {rec_id} eliminado (si existía).")

# --- Página: Histórico ---
elif page == "Histórico":
    st.title("📜 Histórico de Predicciones")
    logs = fetch_logs()
    if logs:
        st.dataframe(pd.DataFrame(logs))
    else:
        st.write("No hay registros aún.")

# --- Página: Explicabilidad ---
elif page == "Explicabilidad":
    st.title("🔍 Explicabilidad de la Predicción")
    if not SHAP_AVAILABLE:
        st.error("Para usar esta sección instala shap y matplotlib:\n\n"
                 "`pip install shap matplotlib`")
    else:
        with st.sidebar.expander("Parámetros (Explicación)", expanded=True):
            inputs_shap = {
                'HEAT_SOURC': st.selectbox("Fuente de calor", ['70','8','22','31','52'], key='e1'),
                'TYPE_MAT'  : st.selectbox("Material", ['Madera','Hormigón','Metal','Plástico'], key='e2'),
                'STRUC_STAT': st.selectbox("Estado estructural", ['Bueno','Regular','Malo'], key='e3'),
                'DETECTOR'  : st.radio("Detector presente", ['Y','N'], key='e4'),
                'DET_TYPE'  : st.selectbox("Tipo de detector", ['1','2','3'], key='e5'),
                'AREA'      : st.slider("Superficie (m²)", 1, 10000, 100, key='e6')
            }
            if st.button("Explicar"):
                pred, proba, df_enc = predict(inputs_shap)
                st.write(f"Predicción: **{pred}**")
                shap_values = explainer.shap_values(df_enc)
                # Usamos gráfico de calor de Streamlit si plt falla
                try:
                    fig = shap.plots.bar(shap_values, df_enc, show=False)
                    st.pyplot(fig)
                except Exception:
                    st.write("No se pudo generar gráfico SHAP. Asegúrate de tener matplotlib.")

# --- Página: Ayuda ---
elif page == "Ayuda":
    st.title("📖 Guías y Manuales")
    with open("Guia de instalacion AIGNITE.pdf","rb") as f:
        st.download_button("📥 Guía de Instalación", f, file_name="Guia_instalacion.pdf")
    with open("Guia de usuario AIGNITE.pdf","rb") as f:
        st.download_button("📥 Guía de Usuario", f, file_name="Guia_usuario.pdf")

# --- Página: Dashboard ---
elif page == "Dashboard":
    st.title("📊 Dashboard Estadísticas")
    logs = fetch_logs()
    if logs:
        df = pd.DataFrame(logs)
        # Distribución
        dist = df['RISK'].value_counts().reindex(['Bajo','Medio','Alto'], fill_value=0)
        st.bar_chart(dist)
        # Serie temporal
        df['timestamp']=pd.to_datetime(df['timestamp'])
        ts = df.set_index('timestamp').resample('D').size()
        st.line_chart(ts)
    else:
        st.write("No hay datos para mostrar.")

# --- Página: Retrain ---
elif page == "Retrain":
    st.title("🔄 Retraining del Modelo")
    if st.button("Ejecutar Retraining"):
        with st.spinner("Entrenando..."):
            import subprocess
            subprocess.run(["python","train_model.py"], check=True)
        st.success("Retraining completado.")

