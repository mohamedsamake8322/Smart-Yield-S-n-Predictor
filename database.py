# === database.py ===

import sqlite3
import pandas as pd
from datetime import datetime

DB_FILE = "history.db"

# ---------- Database Initialization ----------

def init_db():
    """
    Initialize the SQLite database with tables for predictions, observations, and location.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            temperature REAL,
            humidity REAL,
            precipitation REAL,
            ph REAL,
            fertilizer REAL,
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


# ---------- Save Prediction ----------

def save_prediction(username, temperature, humidity, precipitation, ph, fertilizer, predicted_yield):
    """
    Save a prediction record to the database.
    """
    timestamp = datetime.now().isoformat()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            username, temperature, humidity, precipitation, ph, fertilizer, predicted_yield, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, temperature, humidity, precipitation, ph, fertilizer, predicted_yield, timestamp))

    conn.commit()
    conn.close()


# ---------- Load Prediction History ----------

def load_history():
    """
    Load all prediction records from the database as a DataFrame.
    Returns:
        pd.DataFrame: All rows from the predictions table.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
    finally:
        conn.close()
    return df


# ---------- Save Field Observation ----------

def save_observation(name, note):
    """
    Save an observation note into the database.
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


# ---------- Save Field Location ----------

def save_location(lat, lon):
    """
    Save a geographic location (latitude and longitude).
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO field_location (latitude, longitude)
        VALUES (?, ?)
    """, (lat, lon))

    conn.commit()
    conn.close()
import sqlite3

def get_user_predictions(username):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, temperature, humidity, precipitation, pH, fertilizer, predicted_yield, timestamp
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
            "Temperature": row[1],
            "Humidity": row[2],
            "Precipitation": row[3],
            "pH": row[4],
            "Fertilizer": row[5],
            "Predicted Yield": row[6],
            "Timestamp": row[7],
        })

    return results
