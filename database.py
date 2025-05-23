# database.py

import sqlite3
from typing import Dict, List, Optional

DB_PATH = "incendios.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS registros_incendios (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        HEAT_SOURC TEXT,
        TYPE_MAT   TEXT,
        STRUC_STAT TEXT,
        DETECTOR   TEXT,
        DET_TYPE   TEXT,
        AREA       REAL,
        RISK       TEXT,
        prob_bajo  REAL,
        prob_medio REAL,
        prob_alto  REAL,
        timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def log_prediction(inputs: Dict, risk: str, proba: List[float], id_manual: Optional[int] = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if id_manual is not None and id_manual > 0:
        # reemplaza o inserta con ID concreto
        cursor.execute('''
            INSERT OR REPLACE INTO registros_incendios
            (id, HEAT_SOURC, TYPE_MAT, STRUC_STAT, DETECTOR, DET_TYPE, AREA,
             RISK, prob_bajo, prob_medio, prob_alto)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            id_manual,
            inputs["HEAT_SOURC"],
            inputs["TYPE_MAT"],
            inputs["STRUC_STAT"],
            inputs["DETECTOR"],
            inputs["DET_TYPE"],
            inputs["AREA"],
            risk,
            proba[0],
            proba[1],
            proba[2],
        ))
    else:
        # autoincremental
        cursor.execute('''
            INSERT INTO registros_incendios
            (HEAT_SOURC, TYPE_MAT, STRUC_STAT, DETECTOR, DET_TYPE, AREA,
             RISK, prob_bajo, prob_medio, prob_alto)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            inputs["HEAT_SOURC"],
            inputs["TYPE_MAT"],
            inputs["STRUC_STAT"],
            inputs["DETECTOR"],
            inputs["DET_TYPE"],
            inputs["AREA"],
            risk,
            proba[0],
            proba[1],
            proba[2],
        ))
    conn.commit()
    conn.close()
def fetch_logs() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM registros_incendios ORDER BY timestamp DESC")
    cols = [c[0] for c in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    return [dict(zip(cols, row)) for row in rows]
