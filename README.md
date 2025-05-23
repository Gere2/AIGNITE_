AIGNITE – Evaluador de Riesgo de Incendio

AIGNITE es una herramienta completa en Python para:

Preprocesar datos de incidentes de incendio (TXT crudo → CSV intermedios, resumen de nulos, gráfico de distribución).
Entrenar y optimizar un modelo de RandomForest (GridSearchCV) para clasificar niveles de riesgo (“Bajo”, “Medio”, “Alto”).
Evaluar nuevos casos con combinación √(p) de probabilidades cuando hay múltiple material, y registrar resultados en SQLite (con opción de ID manual y detección de duplicados).
Explorar explicaciones locales de cada predicción usando SHAP.
Visualizar métricas y tendencias en un dashboard: distribuciones, series temporales y heatmaps de confusión.
Operar tanto vía interfaz web (Streamlit) como línea de comandos (CLI) según necesidad.
🚀 Instalación

git clone https://github.com/gere2/aignite_.git
cd aignite_
python3 -m venv .venv            # (opcional, recomendado)
source .venv/bin/activate        # Linux / macOS
# o .venv\Scripts\activate       # Windows

pip install -r requirements.txt
🛠️ Uso

1. Interfaz web (Streamlit)
streamlit run app.py
En el navegador encontrarás estas pestañas:

Preprocesado
Sube el fichero crudo fireincident.txt.
Visualiza un DataFrame con recuento y % de valores nulos.
Descarga CSV intermedios: data.csv, data_filtrada.csv, data_menos_nans.csv, data_final.csv.
Barplot de la distribución de FIRE_SPRD.
Evaluar
Introduce parámetros: Fuente de calor, Material combustible (uno o varios), Estado, Detector, Tipo de detector, Área.
Para múltiples materiales, las probabilidades se transforman con √(p) antes de promediar.
Campo ID manual (opcional): si lo rellenas, detecta duplicados y evita sobrescribir.
Guarda cada predicción en SQLite (aignite.db).
CRUD
Consulta y elimina registros por su ID.
Histórico
Filtra por nivel de riesgo y por rango de fechas.
Explora todos los registros guardados.
Explicabilidad
Visualiza gráficos SHAP locales por cada material seleccionado.
Dashboard
Gráficos de barras (conteo por nivel) y líneas (serie temporal de registros).
Retrain
Vuelve a entrenar el modelo desde cero (preprocesado, balanceo, GridSearchCV).
Muestra el classification report y un heatmap de la matriz de confusión.
2. Interfaz de línea de comandos (CLI)
python cli.py
El menú interactivo permite:

Montar Drive (solo en Colab).
Preprocesar datos y exportar CSV/ gráficos.
Entrenar modelo (mismos pasos que Retrain).
Predecir pidiendo inputs por consola.
Consultar, eliminar, listar registros.
Guardar con ID manual, comprobando duplicados.
¿Por qué mantener el CLI?
Facilita automatización en entornos sin GUI (servidores, pipelines).
Arranque más rápido y menor consumo que la app web.
Integración sencilla en scripts legacy o contenedores headless.
📁 Estructura del repositorio

data/
├── raw/fireincident.txt         # Fichero crudo original
├── intermediate/                # CSV generados en Preprocesado
└── reports/                     # Resúmenes de nulos y gráficos

docs/
├── Guia de instalacion AIGNITE.pdf
└── Guia de usuario AIGNITE.pdf

models/
└── aignite_model.pkl            # Bundle (modelo + columnas + cat_cols)

aignite.db                       # Base de datos SQLite

app.py                           # Interfaz web Streamlit
cli.py                           # Interfaz de línea de comandos
database.py                      # CRUD y logging en SQLite
train_model.py                   # Preprocesado + entrenamiento + serialización
requirements.txt                 # Dependencias pip
README.md                        # Este archivo
🤝 Contribuciones

Haz fork y crea una rama (feature/nombre).
Haz commit de tus cambios.
Abre un Pull Request.
