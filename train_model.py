import os
import pandas as pd
from sklearn.ensemble    import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics     import classification_report, confusion_matrix
import joblib

# 1. Carga del raw file (sep="^") y filtrado de las columnas de interés
raw_path = "data/raw/fireincident-2.txt"
df = pd.read_csv(raw_path, sep="^", engine="python", header=0, encoding="latin1")

# Sólo quedarnos con las que usas en tu notebook:
df = df[[
    "HEAT_SOURC",   # Fuente de calor
    "TYPE_MAT",     # Tipo de material
    "STRUC_STAT",   # Estado estructural
    "DETECTOR",     # Presencia de detector
    "DET_TYPE",     # Tipo de detector
    "FIRE_SPRD"     # Grado de propagación (1–5)
]]

# 2. Limpieza: eliminar filas con 'UUU' o NaN
df = df.replace("UUU", pd.NA).dropna()

# 3. Crear variable categórica RISK a partir de FIRE_SPRD
df["FIRE_SPRD"] = df["FIRE_SPRD"].astype(int)
df["RISK"] = pd.cut(
    df["FIRE_SPRD"],
    bins=[0, 2, 3, 5],              # 1–2 → Bajo; 3 → Medio; 4–5 → Alto
    labels=["Bajo", "Medio", "Alto"],
    include_lowest=True
)
# Ya no necesitamos FIRE_SPRD como predictor
df = df.drop(columns=["FIRE_SPRD"])

# 4. Balanceo de clases (1325 muestras de cada una, tal como en tu notebook)
min_count = df["RISK"].value_counts().min()
df_bal = pd.concat([
    df[df["RISK"] == cls].sample(min_count, random_state=42)
    for cls in ["Bajo", "Medio", "Alto"]
], axis=0).reset_index(drop=True)

# 5. Definir variables categóricas y numéricas
cat_cols = ["HEAT_SOURC", "TYPE_MAT", "STRUC_STAT", "DETECTOR", "DET_TYPE"]
num_cols = []  # si tuvieras otras numéricas, las añades aquí (por ej. AREA)

# 6. One‐hot encoding
X = pd.get_dummies(df_bal[cat_cols], columns=cat_cols)
# si hubiera num_cols: X[num_cols] = df_bal[num_cols]
y = df_bal["RISK"]

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. GridSearchCV para Random Forest
param_grid = {
    "n_estimators":    [200, 500],
    "max_depth":       [None, 10, 20],
    "min_samples_leaf":[1, 2, 5]
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("Mejores parámetros:", grid.best_params_)

# 9. Evaluación
y_pred = model.predict(X_test)
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Serializar bundle
os.makedirs("models", exist_ok=True)
bundle = {
    "model":    model,
    "columns":  X_train.columns.tolist(),
    "cat_cols": cat_cols
}
joblib.dump(bundle, "models/aignite_model.pkl")
print("Modelo guardado en models/aignite_model.pkl")
