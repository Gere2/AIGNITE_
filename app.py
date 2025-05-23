import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
import numpy as np
from database import init_db, log_prediction, fetch_logs
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL DE PÁGINA Y ESTILOS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIGNITE – Evaluador de Riesgo de Incendio",
    page_icon="🔥",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* …tu CSS global aquí… */
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# DICCIONARIOS DE MAPEOS
# ─────────────────────────────────────────────────────────────────────────────
TYPE_MAT_MAP = {
    '': 'TYPE MATERIAL FIRST IGNITED',
            '00': 'Type of material first ignited, other',
            '1': 'Flammable Gas',
            '10': 'Flammable gas, other',
            '11': 'Natural gas',
            '12': 'LP gas',
            '13': 'Anesthetic gas',
            '14': 'Acetylene',
            '15': 'Hydrogen',
            '2': 'Flammable, Combustible Liquid',
            '20': 'Flammable or combustible liquid, other',
            '21': 'Ether, pentane type flammable liquid',
            '22': 'JP4 jet fuel & methyl ethyl ketone type flammable',
            '23': 'Gasoline',
            '24': 'Turpentine, butyl alcohol type flammable liquid',
            '25': 'Kerosene, No.1 and 2 fuel oil, diesel type',
            '26': 'Cottonseed oil, creosote oil type combustible',
            '27': 'Cooking oil, transformer or lubricating oil',
            '28': 'Ethanol',
            '3': 'Volatile Solid or Chemical',
            '30': 'Volatile solid or chemical, other',
            '31': 'Fat, grease, butter, margarine, lard',
            '32': 'Petroleum jelly and non-food grease',
            '33': 'Polish, paraffin, wax',
            '34': 'Adhesive, resin, tar, glue, asphalt, pitch',
            '35': 'Paint, varnish - applied',
            '36': 'Combustible metal, included are magnesium',
            '37': 'Solid chemical, included are explosives',
            '38': 'Radioactive material',
            '4': 'Plastics',
            '41': 'Plastic',
            '5': 'Natural Product',
            '50': 'Natural product, other',
            '51': 'Rubber, excluding synthetic rubbers',
            '52': 'Cork',
            '53': 'Leather',
            '54': 'Hay, straw',
            '55': 'Grain, natural fiber, (preprocess)',
            '56': 'Coal, coke, briquettes, peat',
            '57': 'Food, starch, excluding fat and grease Code 31',
            '58': 'Tobacco',
            '6': 'Wood or Paper Processed',
            '60': 'Wood or paper, processed, other',
            '61': 'Wood chips, sawdust, shavings',
            '62': 'Round timber, including round posts, poles',
            '63': 'Sawn wood, including all finished lumber',
            '64': 'Plywood',
            '65': 'Fiberboard, particleboard, and hardboard',
            '66': 'Wood pulp',
            '67': 'Paper, including cellulose, waxed paper',
            '68': 'Cardboard',
            '7': 'Fabric, Textiles, Fur',
            '70': 'Fabric, textile, fur, other',
            '71': 'Fabric, fiber, cotton, blends, rayon, wool',
            '74': 'Fur, silk, other fabric.',
            '75': 'Wig',
            '76': 'Human hair',
            '77': 'Plastic coated fabric',
            '8': 'Material Compounded with Oil',
            '80': 'Material compounded with oil, other',
            '81': 'Linoleum',
            '82': 'Oilcloth',
            '86': 'Asphalt treated material',
            '9': 'Other Material',
            '99': 'Multiple types of material',
            'UU': 'Undetermined'
}

HEAT_SOURC_MAP = {
    '': 'HEAT SOURCE',
            '00': 'Heat source: other',
            '1': 'Operating equipment',
            '10': 'Heat from powered equipment, other',
            '11': 'Spark, ember or flame from operating equipment',
            '12': 'Radiated, conducted heat from operating equipment',
            '13': 'Arcing',
            '4': 'Hot or Smoldering Object',
            '40': 'Hot or smoldering object, other',
            '41': 'Heat, spark from friction',
            '42': 'Molten, hot material',
            '43': 'Hot ember or ash',
            '5': 'Explosives, Fireworks',
            '50': 'Explosive, fireworks, other',
            '51': 'Munitions',
            '53': 'Blasting agent',
            '54': 'Fireworks',
            '55': 'Model and amateur rockets',
            '56': 'Incendiary device',
            '6': 'Other Open Flame or Smoking Materials',
            '60': 'Heat from other open flame or smoking materials',
            '61': 'Cigarette',
            '62': 'Pipe or cigar',
            '63': 'Heat from undetermined smoking material',
            '64': 'Match',
            '65': 'Cigarette lighter',
            '66': 'Candle',
            '67': 'Warning or road flare; fusee',
            '68': 'Backfire from internal combustion engine',
            '69': 'Flame/torch used for lighting',
            '7': 'Chemical, Natural Heat Sources',
            '70': 'Chemical, natural heat source, other',
            '71': 'Sunlight',
            '72': 'Chemical reaction',
            '73': 'Lightning',
            '74': 'Other static discharge',
            '8': 'Heat Spread from Another Fire',
            '80': 'Heat spread from another fire, other',
            '81': 'Heat from direct flame, convection currents',
            '82': 'Radiated heat from another fire',
            '83': 'Flying brand, ember, spark',
            '84': 'Conducted heat from another fire',
            '9': 'Other Heat Sources',
            '97': 'Multiple heat sources including multiple ignitions',
            'UU': 'Undetermined'
}

STRUC_STAT_MAP = {
    '0': 'Other',
    '1': 'Under construction',
    '2': 'In normal use',
    '3': 'Idle, not routinely used',
    '4': 'Under major renovation',
    '5': 'Vacant and secured',
    '6': 'Vacant and unsecured',
    '7': 'Being demolished',
    'U': 'Undetermined'
}

DETECTOR_MAP = {
    '1': 'Detectors Present',
    'N': 'None Present',
    'Y': 'Detectors Present',
    'U': 'Undetermined'
}

DET_TYPE_MAP = {
    '0': 'Other',
    '1': 'Smoke',
    '2': 'Heat',
    '3': 'Combination smoke - heat',
    '4': 'Sprinkler, water flow detection',
    '5': 'More than 1 type present',
    'U': 'Undetermined'
}



def validar_codigo(inputs: dict) -> (bool, list):
    errores = []
    if inputs["HEAT_SOURC"] not in HEAT_SOURC_MAP:
        errores.append(f"Fuente de calor inválida: {inputs['HEAT_SOURC']}")
    if inputs["TYPE_MAT"] not in TYPE_MAT_MAP:
        errores.append(f"Material combustible inválido: {inputs['TYPE_MAT']}")
    if inputs["STRUC_STAT"] not in STRUC_STAT_MAP:
        errores.append(f"Estado estructural inválido: {inputs['STRUC_STAT']}")
    if inputs["DETECTOR"] not in DETECTOR_MAP:
        errores.append(f"Detector inválido: {inputs['DETECTOR']}")
    if inputs["DET_TYPE"] not in DET_TYPE_MAP:
        errores.append(f"Tipo de detector inválido: {inputs['DET_TYPE']}")
    return (len(errores) == 0, errores)

# ─────────────────────────────────────────────────────────────────────────────
# 4) CARGA DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load("models/aignite_model.pkl")
    return bundle["model"], bundle["columns"], bundle["cat_cols"]

clf, model_columns, cat_cols = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# 5) INTEGRACIÓN SHAP
# ─────────────────────────────────────────────────────────────────────────────
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True

    @st.cache_resource
    def load_explainer():
        return shap.TreeExplainer(clf)

    explainer = load_explainer()

except ImportError:
    SHAP_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 6) FUNCIÓN DE PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────
def predict(data: dict):
    df = pd.DataFrame([data])
    for c in cat_cols:
        df[c] = df[c].astype(str)
    df_enc = pd.get_dummies(df, columns=cat_cols).reindex(
        columns=model_columns, fill_value=0
    )
    pred  = clf.predict(df_enc)[0]
    proba = clf.predict_proba(df_enc)[0]
    return pred, proba, df_enc

RISK_STYLE = {
    "Bajo":  {"func": st.success, "icon": "🟢"},
    "Medio": {"func": st.warning, "icon": "🟡"},
    "Alto":  {"func": st.error,   "icon": "🔴"}
}

def main():
    # ─────────────────────────────────────────────────────────────────────────
    # Inicializar base de datos y cargar modelo
    # ─────────────────────────────────────────────────────────────────────────
    init_db()

    @st.cache_resource
    def load_explainer():
        return shap.TreeExplainer(clf)

    explainer = load_explainer()

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

    RISK_STYLE = {
        "Bajo":  {"func": st.success, "icon": "🟢"},
        "Medio": {"func": st.warning, "icon": "🟡"},
        "Alto":  {"func": st.error,   "icon": "🔴"}
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Navegación
    # ─────────────────────────────────────────────────────────────────────────
    page = st.sidebar.radio("🏠 Navegación", [
        "Evaluar", "CRUD", "Histórico",
        "Explicabilidad", "Ayuda",
        "Dashboard", "Retrain"
    ])
    # ─────────────────────────────────────────────────────────────────────────
    # Página: Evaluar
    # ─────────────────────────────────────────────────────────────────────────
    if page == "Evaluar":
        st.markdown("## <span class='emoji'>🔥</span> Evaluar Riesgo", unsafe_allow_html=True)
        st.write(
            "Selecciona los parámetros, opcionalmente el ID, y haz clic en **🔥 Evaluar riesgo**. Verás el resultado a la derecha.")

        col_inputs, col_results = st.columns([1, 2], gap="large")

        with col_inputs:
            st.markdown("### 🔧 Parámetros de evaluación")

            # Fuente de calor
            heat_val = st.selectbox(
                "Fuente de calor",
                list(HEAT_SOURC_MAP.keys()),
                key="eval_heat"
            )
            st.caption(f"📖 {HEAT_SOURC_MAP[heat_val]}")

            # Material combustible (multiselect)
            mat_vals = st.multiselect(
                "Material combustible",
                options=list(TYPE_MAT_MAP.keys()),
                default=[list(TYPE_MAT_MAP.keys())[1]],
                key="eval_mat"
            )
            if mat_vals:
                st.caption("📖 " + ", ".join(TYPE_MAT_MAP[m] for m in mat_vals))
            else:
                st.caption("📖 Selecciona al menos un material")

            # Estado estructural
            struct_val = st.selectbox(
                "Estado estructural",
                list(STRUC_STAT_MAP.keys()),
                key="eval_struct"
            )
            st.caption(f"📖 {STRUC_STAT_MAP[struct_val]}")

            # Detector presente
            det_val = st.selectbox(
                "Detector presente",
                list(DETECTOR_MAP.keys()),
                key="eval_det"
            )
            st.caption(f"📖 {DETECTOR_MAP[det_val]}")

            # Tipo de detector
            dtype_val = st.selectbox(
                "Tipo de detector",
                list(DET_TYPE_MAP.keys()),
                key="eval_dtype"
            )
            st.caption(f"📖 {DET_TYPE_MAP[dtype_val]}")

            # Área
            area_val = st.slider(
                "Superficie (m²)",
                min_value=1,
                max_value=10000,
                value=100,
                key="eval_area"
            )

            # ID manual (opcional)
            id_manual = st.number_input(
                "ID manual (opcional)",
                min_value=0,
                value=0,
                step=1,
                help="Si lo dejas a 0 se usará autoincremental",
                key="eval_id"
            )

            evaluate = st.button("🔥 Evaluar riesgo", key="eval_btn")

        with col_results:
            if not evaluate:
                st.info("Pulsa «🔥 Evaluar riesgo» para ver el resultado aquí.")
            else:
                # Preparamos inputs
                inputs_eval = {
                    "HEAT_SOURC": heat_val,
                    "TYPE_MAT": mat_vals[0] if mat_vals else None,
                    "STRUC_STAT": struct_val,
                    "DETECTOR": det_val,
                    "DET_TYPE": dtype_val,
                    "AREA": area_val
                }

                # Validación
                ok, errores = validar_codigo(inputs_eval)
                if not mat_vals:
                    col_results.error("Debes seleccionar al menos un material combustible.")
                elif not ok:
                    for err in errores:
                        col_results.error(err)
                else:
                    # Si hay varios materiales, promediamos probas
                    import numpy as np
                    probas = []
                    for m in mat_vals:
                        inp = {**inputs_eval, "TYPE_MAT": m}
                        _, p, _ = predict(inp)
                        probas.append(p)
                    avg_proba = np.mean(probas, axis=0)

                    # Decidimos la clase final
                    risk = clf.classes_[np.argmax(avg_proba)]

                    # Logueamos, guardando todos los mats como CSV
                    log_prediction(
                        {**inputs_eval, "TYPE_MAT": ",".join(mat_vals)},
                        risk,
                        avg_proba,
                        id_manual if id_manual > 0 else None
                    )

                    # Mostrar resultado
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    style = RISK_STYLE[risk]
                    style["func"](f"{style['icon']} Nivel de riesgo: **{risk}**")
                    st.write(
                        f"🟢 Bajo: {avg_proba[0]:.1%}  |  "
                        f"🟡 Medio: {avg_proba[1]:.1%}  |  "
                        f"🔴 Alto: {avg_proba[2]:.1%}"
                    )
                    if risk == "Bajo":
                        st.balloons()
                    st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Página: CRUD
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "CRUD":
        st.markdown("## <span class='emoji'>🔧</span> CRUD de Registros", unsafe_allow_html=True)
        rec_id = st.number_input("ID del registro", min_value=1, step=1)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Consultar"):
                rec = next((r for r in fetch_logs() if r["id"] == rec_id), None)
                if rec:
                    st.json(rec)
                else:
                    st.warning("Registro no encontrado.")
        with col2:
            if st.button("Eliminar"):
                conn = sqlite3.connect("incendios.db")
                c = conn.cursor()
                c.execute("DELETE FROM registros_incendios WHERE id = ?", (rec_id,))
                conn.commit()
                conn.close()
                st.info(f"Registro {rec_id} eliminado (si existía).")

    # ─────────────────────────────────────────────────────────────────────────
    # Página: Histórico
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "Histórico":
        st.markdown("## <span class='emoji'>📜</span> Histórico de Predicciones", unsafe_allow_html=True)
        logs = fetch_logs()
        if not logs:
            st.write("No hay registros aún.")
        else:
            # Montamos el DataFrame y convertimos timestamp
            df = pd.DataFrame(logs)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Sidebar de filtros
            with st.sidebar.expander("🔍 Filtros del Historial", expanded=True):
                # Filtrar por nivel de riesgo
                niveles = st.multiselect(
                    "Niveles de riesgo",
                    options=["Bajo", "Medio", "Alto"],
                    default=["Bajo", "Medio", "Alto"]
                )
                # Filtrar por rango de fechas
                fechas = st.date_input(
                    "Rango de fechas",
                    value=[df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()]
                )

            # Aplicamos los filtros
            mask = (
                    df["RISK"].isin(niveles)
                    & (df["timestamp"].dt.date >= fechas[0])
                    & (df["timestamp"].dt.date <= fechas[1])
            )
            df_filtrado = df.loc[mask].sort_values("timestamp", ascending=False)

            # Mostramos resultado
            st.write(f"Mostrando {len(df_filtrado)} registros:")
            st.dataframe(df_filtrado.reset_index(drop=True))

    # ─────────────────────────────────────────────────────────────────────────
    # --- Página: Explicabilidad ---
    elif page == "Explicabilidad":
        st.markdown("## <span class='emoji'>🔍</span> Explicabilidad de la Predicción", unsafe_allow_html=True)
        st.write(
            "Selecciona los parámetros y haz clic en **🔎 Explicar riesgo**. Verás a la derecha la importancia global y, si SHAP está disponible, el gráfico local.")

        col_inputs, col_results = st.columns([1, 2], gap="large")

        with col_inputs:
            st.markdown("### 🔧 Parámetros de explicación")
            heat = col_inputs.selectbox("Fuente de calor", list(HEAT_SOURC_MAP.keys()), key="ex1")
            col_inputs.caption(f"📖 {HEAT_SOURC_MAP.get(heat, '')}")

            mat = col_inputs.selectbox("Material combustible", list(TYPE_MAT_MAP.keys()), key="ex2")
            col_inputs.caption(f"📖 {TYPE_MAT_MAP.get(mat, '')}")

            struct = col_inputs.selectbox("Estado estructural", list(STRUC_STAT_MAP.keys()), key="ex3")
            col_inputs.caption(f"📖 {STRUC_STAT_MAP.get(struct, '')}")

            det = col_inputs.selectbox("Detector presente", list(DETECTOR_MAP.keys()), key="ex4")
            col_inputs.caption(f"📖 {DETECTOR_MAP.get(det, '')}")

            dtype = col_inputs.selectbox("Tipo de detector", list(DET_TYPE_MAP.keys()), key="ex5")
            col_inputs.caption(f"📖 {DET_TYPE_MAP.get(dtype, '')}")

            area = col_inputs.slider("Superficie (m²)", 1, 10000, 100, key="ex6")

            explain = col_inputs.button("🔎 Explicar riesgo")

        with col_results:
            if not explain:
                col_results.info("Pulsa «🔎 Explicar riesgo» para ver la explicación aquí.")
            else:
                inputs_exp = {
                    "HEAT_SOURC": heat,
                    "TYPE_MAT": mat,
                    "STRUC_STAT": struct,
                    "DETECTOR": det,
                    "DET_TYPE": dtype,
                    "AREA": area
                }
                ok, errores = validar_codigo(inputs_exp)
                if not ok:
                    for err in errores:
                        col_results.error(err)
                else:
                    # 1) Importancia global
                    feat_imp = pd.Series(clf.feature_importances_, index=model_columns) \
                        .sort_values(ascending=False).head(10)
                    col_results.markdown("### 📊 Importancia global (Random Forest)")
                    col_results.bar_chart(feat_imp)

                    # 2) Explicación local con SHAP
                    if not SHAP_AVAILABLE:
                        col_results.error("Instala `shap` y `matplotlib` para ver la explicación local.")
                    else:
                        # Preparamos la fila codificada
                        _, _, df_enc = predict(inputs_exp)
                        shap_values = explainer.shap_values(df_enc)
                        col_results.markdown("### 🔍 Explicación local (SHAP)")
                        # Bar chart SHAP
                        fig, ax = plt.subplots()
                        shap.plots.bar(shap_values, df_enc, show=False, ax=ax)
                        col_results.pyplot(fig)
    # ─────────────────────────────────────────────────────────────────────────
    # Página: Ayuda
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "Ayuda":
        st.markdown("## <span class='emoji'>📖</span> Guías y Manuales", unsafe_allow_html=True)
        st.write("Descarga los documentos de instalación y usuario para AIGNITE.")
        docs = {
            "🛠️ Guía de Instalación":   "Guia_Instalacion_AIGNITE.pdf",
            "📄 Guía de Usuario":       "Guia_Usuario_AIGNITE.pdf"
        }
        cols = st.columns(len(docs), gap="large")
        for (label, filename), col in zip(docs.items(), cols):
            file_path = os.path.join("docs", filename)
            with col:
                st.markdown(f"#### {label}")
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="📥 Descargar",
                            data=f,
                            file_name=filename,
                            use_container_width=True
                        )
                else:
                    st.error(f"{label} no encontrada en `/docs/{filename}`")

    # ─────────────────────────────────────────────────────────────────────────
    # Página: Dashboard
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "Dashboard":
        st.markdown("## <span class='emoji'>📊</span> Dashboard Estadísticas", unsafe_allow_html=True)
        logs = fetch_logs()
        if logs:
            df = pd.DataFrame(logs)
            dist = df["RISK"].value_counts().reindex(["Bajo","Medio","Alto"], fill_value=0)
            st.bar_chart(dist)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            ts = df.set_index("timestamp").resample("D").size()
            st.line_chart(ts)
        else:
            st.write("No hay datos para mostrar.")

    # ─────────────────────────────────────────────────────────────────────────
    # Página: Retrain
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "Retrain":
        st.markdown("## <span class='emoji'>🔄</span> Retraining del Modelo", unsafe_allow_html=True)
        if st.button("Ejecutar Retraining"):
            with st.spinner("Entrenando..."):
                # Asumimos que train_model.py define:
                # def retrain_and_return_test():
                #     model, X_test, y_test = ...  # entrena, guarda modelo, y devuelve test
                from train_model import retrain_and_return_test
                model, X_test, y_test = retrain_and_return_test()

            st.success("Retraining completado.")

            # Predicción sobre el conjunto de test
            y_pred = model.predict(X_test)

            # 1) Reporte de clasificación
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report_dict).transpose()
            st.markdown("#### 📋 Reporte de clasificación")
            st.dataframe(df_report)

            # 2) Matriz de confusión
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            df_cm = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
            st.markdown("#### 🔢 Matriz de confusión")
            st.table(df_cm)

if __name__ == "__main__":
    main()
