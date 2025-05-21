import sqlite3
import pandas as pd
from datetime import datetime

DB_FILE = "history.db"

# === 🔹 Initialisation de la base de données ===
def init_db():
    """
    Initialise la base de données SQLite avec les tables nécessaires.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            features TEXT,
            predicted_yield REAL,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            note TEXT,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS field_location (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL
        )
    """)

    conn.commit()
    conn.close()

# === 🔹 Enregistrer une prédiction ===
def save_prediction(username, features, predicted_yield):
    """
    Enregistre une prédiction dans la base de données.
    """
    timestamp = datetime.now().isoformat()
    features_str = ",".join(map(str, features))  # 🔹 Convertir `features` en texte

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (username, features, predicted_yield, timestamp)
        VALUES (?, ?, ?, ?)
    """, (username, features_str, predicted_yield, timestamp))

    conn.commit()
    conn.close()

# === 🔹 Charger l'historique des prédictions ===
def load_history():
    """
    Charge toutes les prédictions depuis la base de données sous forme de DataFrame.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
    finally:
        conn.close()
    return df

# === 🔹 Enregistrer une observation ===
def save_observation(name, note):
    """
    Enregistre une observation dans la base de données.
    """
    timestamp = datetime.now().isoformat()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO observations (name, note, timestamp)
        VALUES (?, ?, ?)
    """, (name, note, timestamp))

    conn.commit()
    conn.close()

# === 🔹 Enregistrer la localisation ===
def save_location(lat, lon):
    """
    Enregistre une localisation géographique.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO field_location (latitude, longitude)
        VALUES (?, ?)
    """, (lat, lon))

    conn.commit()
    conn.close()

# === 🔹 Récupérer les prédictions d'un utilisateur ===
def get_user_predictions(username):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, features, predicted_yield, timestamp
        FROM predictions
        WHERE username = ?
        ORDER BY timestamp DESC
    """, (username,))
    
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "Username": row[0],
            "Features": row[1].split(","),  # 🔹 Convertir `features` en liste
            "Predicted Yield": row[2],
            "Timestamp": row[3],
        })

    return results
