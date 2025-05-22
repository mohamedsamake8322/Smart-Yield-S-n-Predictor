import sqlite3
import pandas as pd
from datetime import datetime
import logging

# 📌 Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_FILE = "history.db"

# === 🔹 Initialisation de la base de données ===
def init_db():
    """
    Initialise la base de données SQLite avec les tables nécessaires.
    """
    try:
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
        logging.info("✅ Base de données initialisée avec succès !")
    except sqlite3.Error as e:
        logging.error(f"🚨 Erreur lors de l'initialisation de la base de données : {e}")
    finally:
        conn.close()

# === 🔹 Enregistrer une prédiction ===
def save_prediction(username, features, predicted_yield):
    """
    Enregistre une prédiction dans la base de données.
    """
    timestamp = datetime.now().isoformat()
    features_str = ",".join(map(str, features))  # 🔹 Convertir `features` en texte

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions (username, features, predicted_yield, timestamp)
            VALUES (?, ?, ?, ?)
        """, (username, features_str, predicted_yield, timestamp))

        conn.commit()
        logging.info(f"✅ Prédiction enregistrée pour {username}.")
    except sqlite3.Error as e:
        logging.error(f"🚨 Erreur lors de l'enregistrement de la prédiction : {e}")
    finally:
        conn.close()

# === 🔹 Charger l'historique des prédictions ===
def load_history():
    """
    Charge toutes les prédictions depuis la base de données sous forme de DataFrame.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        logging.info("✅ Historique des prédictions chargé avec succès.")
        return df
    except sqlite3.Error as e:
        logging.error(f"🚨 Erreur lors du chargement de l'historique : {e}")
        return pd.DataFrame()  # 🔹 Retourner un DataFrame vide en cas d'erreur
    finally:
        conn.close()

# === 🔹 Enregistrer une observation ===
def save_observation(name, note):
    """
    Enregistre une observation dans la base de données.
    """
    timestamp = datetime.now().isoformat()
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO observations (name, note, timestamp)
            VALUES (?, ?, ?)
        """, (name, note, timestamp))

        conn.commit()
        logging.info(f"✅ Observation enregistrée : {name}.")
    except sqlite3.Error as e:
        logging.error(f"🚨 Erreur lors de l'enregistrement de l'observation : {e}")
    finally:
        conn.close()

# === 🔹 Enregistrer la localisation ===
def save_location(lat, lon):
    """
    Enregistre une localisation géographique.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO field_location (latitude, longitude)
            VALUES (?, ?)
        """, (lat, lon))

        conn.commit()
        logging.info(f"✅ Localisation enregistrée : ({lat}, {lon}).")
    except sqlite3.Error as e:
        logging.error(f"🚨 Erreur lors de l'enregistrement de la localisation : {e}")
    finally:
        conn.close()

# === 🔹 Récupérer les prédictions d'un utilisateur ===
def get_user_predictions(username):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT username, features, predicted_yield, timestamp
            FROM predictions
            WHERE username = ?
            ORDER BY timestamp DESC
        """, (username,))
        
        rows = cursor.fetchall()
        logging.info(f"✅ Prédictions récupérées pour {username}.")
        
        results = [
            {"Username": row[0], "Features": row[1].split(","), "Predicted Yield": row[2], "Timestamp": row[3]}
            for row in rows
        ]
        return results
    except sqlite3.Error as e:
        logging.error(f"🚨 Erreur lors de la récupération des prédictions : {e}")
        return []
    finally:
        conn.close()
