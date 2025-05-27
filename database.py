import sqlite3
import pandas as pd
from datetime import datetime
import logging
from flask_jwt_extended import jwt_required, get_jwt_identity

# 🔹 Logger configuration (utilisation d'un logger spécifique pour `database.py`)
logger = logging.getLogger(__name__)

DB_FILE = "history.db"

# === Initialize Database ===
def init_db():
    """ Initializes SQLite database with required tables. """
    try:
        with sqlite3.connect(DB_FILE) as conn:
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
            logger.info("✅ Database initialized successfully!")
    except sqlite3.Error as e:
        logger.error(f"🚨 Error initializing database: {e}")

# === Save Prediction (JWT Required) ===
@jwt_required()
def save_prediction(features, predicted_yield):
    """ Saves a prediction to the database securely. """
    username = get_jwt_identity()
    timestamp = datetime.now().isoformat()
    features_str = ",".join(map(str, features))

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (username, features, predicted_yield, timestamp)
                VALUES (?, ?, ?, ?)
            """, (username, features_str, predicted_yield, timestamp))
            logger.info(f"✅ Prediction saved for {username}.")
    except sqlite3.Error as e:
        logger.error(f"🚨 Error saving prediction: {e}")

# === Load Prediction History (JWT Required) ===
@jwt_required()
def get_user_predictions():
    """ Loads all predictions for the authenticated user. """
    username = get_jwt_identity()

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT features, predicted_yield, timestamp
                FROM predictions
                WHERE username = ?
                ORDER BY timestamp DESC
            """, (username,))
            
            rows = cursor.fetchall()
            logger.info(f"✅ Predictions retrieved for {username}.")

            return [
                {"Features": row[0].split(","), "Predicted Yield": row[1], "Timestamp": row[2]}
                for row in rows
            ]
    except sqlite3.Error as e:
        logger.error(f"🚨 Error retrieving predictions: {e}")
        return []

# === Save Observation ===
def save_observation(name, note):
    """ Saves an observation to the database. """
    timestamp = datetime.now().isoformat()
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO observations (name, note, timestamp)
                VALUES (?, ?, ?)
            """, (name, note, timestamp))
            logger.info(f"✅ Observation saved: {name}.")
    except sqlite3.Error as e:
        logger.error(f"🚨 Error saving observation: {e}")

# === Save Location ===
def save_location(lat, lon):
    """ Saves a geographic location to the database. """
    timestamp = datetime.now().isoformat()
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO field_location (latitude, longitude, timestamp)
                VALUES (?, ?, ?)
            """, (lat, lon, timestamp))
            logger.info(f"✅ Location saved: ({lat}, {lon}).")
    except sqlite3.Error as e:
        logger.error(f"🚨 Error saving location: {e}")
        print("Exécution terminée avec succès !")