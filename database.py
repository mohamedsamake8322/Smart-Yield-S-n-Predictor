import sqlite3
import pandas as pd
from datetime import datetime
import logging

# 🔹 Logger Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_FILE = "history.db"

# === Initialize Database ===
def init_db():
    """ Initializes SQLite database with required tables. """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                features TEXT NOT NULL,
                predicted_yield REAL NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                note TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS field_location (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                timestamp TEXT NOT NULL
            );
        """)

        conn.commit()
        logging.info("✅ Database initialized successfully!")
    except sqlite3.Error as e:
        logging.error(f"🚨 Error initializing database: {e}")
    finally:
        conn.close()

# === Save Prediction ===
def save_prediction(username, features, predicted_yield):
    """ Saves a prediction to the database. """
    timestamp = datetime.now().isoformat()
    features_str = ",".join(map(str, features))

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (username, features, predicted_yield, timestamp)
            VALUES (?, ?, ?, ?)
        """, (username, features_str, predicted_yield, timestamp))
        conn.commit()
        logging.info(f"✅ Prediction saved for {username}.")
    except sqlite3.Error as e:
        logging.error(f"🚨 Error saving prediction: {e}")
    finally:
        conn.close()

# === Load Prediction History ===
def load_history():
    """ Loads all predictions from the database as a DataFrame. """
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        logging.info("✅ Prediction history loaded successfully.")
        return df
    except sqlite3.Error as e:
        logging.error(f"🚨 Error loading history: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# === Save Observation ===
def save_observation(name, note):
    """ Saves an observation to the database. """
    timestamp = datetime.now().isoformat()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO observations (name, note, timestamp)
            VALUES (?, ?, ?)
        """, (name, note, timestamp))
        conn.commit()
        logging.info(f"✅ Observation saved: {name}.")
    except sqlite3.Error as e:
        logging.error(f"🚨 Error saving observation: {e}")
    finally:
        conn.close()

# === Save Location ===
def save_location(lat, lon):
    """ Saves a geographic location to the database. """
    timestamp = datetime.now().isoformat()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO field_location (latitude, longitude, timestamp)
            VALUES (?, ?, ?)
        """, (lat, lon, timestamp))
        conn.commit()
        logging.info(f"✅ Location saved: ({lat}, {lon}).")
    except sqlite3.Error as e:
        logging.error(f"🚨 Error saving location: {e}")
    finally:
        conn.close()

# === Retrieve User Predictions ===
def get_user_predictions(username):
    """ Retrieves all predictions for a specific user. """
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
        logging.info(f"✅ Predictions retrieved for {username}.")
        
        results = [
            {"Username": row[0], "Features": row[1].split(","), "Predicted Yield": row[2], "Timestamp": row[3]}
            for row in rows
        ]
        return results
    except sqlite3.Error as e:
        logging.error(f"🚨 Error retrieving predictions: {e}")
        return []
    finally:
        conn.close()
