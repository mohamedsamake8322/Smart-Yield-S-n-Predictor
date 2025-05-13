# === database.py ===
import sqlite3
from datetime import datetime

DB_FILE = "history.db"

# Initialize database with predictions and observations
def init_db():
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
    conn.commit()
    conn.close()

# Save prediction with username
def save_prediction(username, temperature, humidity, precipitation, ph, fertilizer, predicted_yield):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO predictions (
        username, temperature, humidity, precipitation, ph, fertilizer, predicted_yield, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, temperature, humidity, precipitation, ph, fertilizer, predicted_yield, timestamp))
    conn.commit()
    conn.close()

# Load full history
def load_history():
    conn = sqlite3.connect(DB_FILE)
    df = None
    try:
        df = conn.execute("SELECT * FROM predictions").fetchall()
    finally:
        conn.close()
    return df

# Save field observation
def save_observation(name, note):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO observations (name, note, timestamp)
    VALUES (?, ?, ?)
    """, (name, note, timestamp))
    conn.commit()
    conn.close()
import sqlite3

def save_location(lat, lon):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS field_location (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL
        )
    """)
    cursor.execute("INSERT INTO field_location (latitude, longitude) VALUES (?, ?)", (lat, lon))
    conn.commit()
    conn.close()
