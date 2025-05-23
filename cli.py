#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import joblib
import math
from database import init_db, log_prediction, fetch_logs, guardar_en_bd_con_id_manual

def montar_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        pass

def preprocesar():
    raw_path = os.path.join("data", "raw", "fireincident.txt")
    df = pd.read_csv(raw_path, sep="^", engine="python", header=0, encoding="latin1")

    # Resumen de nulos
    null_summary = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_pct': df.isna().mean() * 100
    })
    os.makedirs("data/reports", exist_ok=True)
    null_summary.to_csv("data/reports/null_summary.csv", index=True)
    print("Resumen de nulos guardado en data/reports/null_summary.csv")

    # CSV intermedios
    os.makedirs("data/intermediate", exist_ok=True)
    df.to_csv("data/intermediate/data.csv", index=False)
    cols = ["HEAT_SOURC","TYPE_MAT","STRUC_STAT","DETECTOR","DET_TYPE","FIRE_SPRD"]
    df[cols].to_csv("data/intermediate/data_filtrada.csv", index=False)
    df_filtrada = df[cols].dropna()
    df_filtrada.to_csv("data/intermediate/data_menos_nans.csv", index=False)
    df_filtrada.to_csv("data/intermediate/data_final.csv", index=False)
    print("CSV intermedios guardados en data/intermediate/")

    # Distribución FIRE_SPRD
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        dist = df["FIRE_SPRD"].value_counts().sort_index()
        plt.figure()
        sns.barplot(x=dist.index, y=dist.values)
        plt.title("Distribución de FIRE_SPRD")
        plt.savefig("data/reports/fire_spread_distribution.png")
        print("Gráfico de distribución guardado en data/reports/fire_spread_distribution.png")
    except ImportError:
        print("Seaborn o matplotlib no están instalados: omitiendo gráfico.")

def entrenar():
    from train_model import retrain_and_return_test
    model, X_test, y_test, best_params = retrain_and_return_test()
    print("Mejores parámetros:", best_params)

def predict_cli():
    bundle = joblib.load("models/aignite_model.pkl")
    clf = bundle["model"]
    cat_cols = bundle["cat_cols"]
    columns = bundle["columns"]

    inputs = {}
    inputs["HEAT_SOURC"] = input("Fuente de calor: ")
    inputs["TYPE_MAT"] = input("Material combustible: ")
    inputs["STRUC_STAT"] = input("Estado estructural: ")
    inputs["DETECTOR"] = input("Detector presente: ")
    inputs["DET_TYPE"] = input("Tipo de detector: ")
    inputs["AREA"] = float(input("Superficie (m²): "))

    df = pd.DataFrame([inputs])
    for c in cat_cols:
        df[c] = df[c].astype(str)
    df_enc = pd.get_dummies(df, columns=cat_cols).reindex(columns=columns, fill_value=0)

    pred = clf.predict(df_enc)[0]
    proba = clf.predict_proba(df_enc)[0]
    print("\nPredicción:", pred)
    print(f"Bajo: {proba[0]:.1%}, Medio: {proba[1]:.1%}, Alto: {proba[2]:.1%}")

def consultar_cli():
    rec_id = int(input("ID del registro: "))
    logs = fetch_logs()
    rec = next((r for r in logs if r["id"] == rec_id), None)
    if rec:
        print("\n", rec)
    else:
        print("Registro no encontrado.")

def eliminar_cli():
    rec_id = int(input("ID del registro a eliminar: "))
    conn = sqlite3.connect("incendios.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM registros_incendios WHERE id=?", (rec_id,))
    conn.commit()
    conn.close()
    print(f"Registro {rec_id} eliminado.")

def listar_cli():
    logs = fetch_logs()
    for rec in logs:
        print(rec)

def guardar_manual_cli():
    rec_id = int(input("ID manual para guardar: "))
    bundle = joblib.load("models/aignite_model.pkl")
    clf = bundle["model"]
    cat_cols = bundle["cat_cols"]
    columns = bundle["columns"]

    inputs = {}
    inputs["HEAT_SOURC"] = input("Fuente de calor: ")
    inputs["TYPE_MAT"] = input("Material combustible: ")
    inputs["STRUC_STAT"] = input("Estado estructural: ")
    inputs["DETECTOR"] = input("Detector presente: ")
    inputs["DET_TYPE"] = input("Tipo de detector: ")
    inputs["AREA"] = float(input("Superficie (m²): "))

    df = pd.DataFrame([inputs])
    for c in cat_cols:
        df[c] = df[c].astype(str)
    df_enc = pd.get_dummies(df, columns=cat_cols).reindex(columns=columns, fill_value=0)

    pred = clf.predict(df_enc)[0]
    proba = clf.predict_proba(df_enc)[0]
    ok = guardar_en_bd_con_id_manual(rec_id, inputs, pred, proba)
    if ok:
        print(f"Registro con ID {rec_id} guardado correctamente.")
    else:
        print(f"Ya existe un registro con el ID {rec_id}. No se ha guardado.")

def menu_principal():
    init_db()
    while True:
        print("\n--- Menú ---")
        print("1. Preprocesar datos")
        print("2. Entrenar modelo")
        print("3. Predecir")
        print("4. Consultar registro")
        print("5. Eliminar registro")
        print("6. Listar registros")
        print("7. Guardar con ID manual")
        print("0. Salir")
        opt = input("Opción: ")
        if opt == "1":
            montar_drive()
            preprocesar()
        elif opt == "2":
            entrenar()
        elif opt == "3":
            predict_cli()
        elif opt == "4":
            consultar_cli()
        elif opt == "5":
            eliminar_cli()
        elif opt == "6":
            listar_cli()
        elif opt == "7":
            guardar_manual_cli()
        elif opt == "0":
            break
        else:
            print("Opción inválida. Intenta de nuevo.")

if __name__ == "__main__":
    menu_principal()
