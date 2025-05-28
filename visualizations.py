# === visualizations.py ===
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
import streamlit as st
import folium
import sqlite3
import requests
import numpy as np
from streamlit_folium import st_folium
from folium.plugins import HeatMap

print("🚀 Script visualizations.py started...")

# 🌍 Definition of agricultural fields
FIELDS = [
    {"name": "Field A", "lat": 12.64, "lon": -8.0},
    {"name": "Field B", "lat": 12.66, "lon": -7.98},
    {"name": "Field C", "lat": 12.63, "lon": -8.02},
]

# 📊 Store data in SQLite
def create_database():
    conn = sqlite3.connect("fields_data.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fields (
        id INTEGER PRIMARY KEY,
        name TEXT,
        latitude REAL,
        longitude REAL,
        temperature REAL,
        humidity REAL,
        stress_level REAL
    )
    """)

    for field in FIELDS:
        cursor.execute("""
        INSERT INTO fields (name, latitude, longitude, temperature, humidity, stress_level)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (field["name"], field["lat"], field["lon"], 27, 60, 0.75))

    conn.commit()
    conn.close()
    print("✅ Data saved in SQLite!")

# 🌍 Generate Folium map
def generate_map():
    m = folium.Map(location=[12.64, -8.0], zoom_start=12)

    for field in FIELDS:
        folium.Marker(
            location=[field["lat"], field["lon"]],
            popup=f"{field['name']} - Stress Level: 0.75",
            icon=folium.Icon(color="green")
        ).add_to(m)

    m.save("fields_map.html")
    print("✅ Map saved as fields_map.html!")

# 🌦️ Retrieve weather data and update the database
API_KEY = "YOUR_API_KEY"

def get_weather_data(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    return response.json()

def update_weather():
    conn = sqlite3.connect("fields_data.db")
    cursor = conn.cursor()

    for field in FIELDS:
        weather = get_weather_data(field["lat"], field["lon"])
        temperature = weather["main"]["temp"]
        humidity = weather["main"]["humidity"]
        stress_level = min(1, max(0, 0.5 + (temperature - 25) * 0.02))

        cursor.execute("UPDATE fields SET temperature=?, humidity=?, stress_level=? WHERE name=?",
                       (temperature, humidity, stress_level, field["name"]))

    conn.commit()
    conn.close()
    print("✅ Weather data updated!")

# 📊 Visualization of crop yields
def plot_yield_distribution(df):
    if "PredictedYield" not in df.columns:
        raise ValueError("❌ Column 'PredictedYield' is missing in DataFrame")

    fig, ax = plt.subplots()
    sns.histplot(df["PredictedYield"], bins=20, kde=True, color="green", ax=ax)
    ax.set_xlabel("Yield (tons/ha)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

def plot_yield_pie(df):
    if "PredictedYield" not in df.columns:
        raise ValueError("❌ Column 'PredictedYield' is missing in DataFrame")

    bins = [0, 10, 20, 30, 50, 100]
    labels = ["<10", "10–20", "20–30", "30–50", ">50"]
    df["yield_bin"] = pd.cut(df["PredictedYield"], bins=bins, labels=labels, right=False)

    counts = df["yield_bin"].value_counts().sort_index()
    colors = sns.color_palette("pastel")

    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.set_title("🎂 Predicted Yield Distribution (Pie Chart)")
    fig.tight_layout()
    return fig

def plot_yield_over_time(df):
    if "timestamp" not in df.columns or "PredictedYield" not in df.columns:
        raise ValueError("❌ Columns 'timestamp' and 'PredictedYield' are required in DataFrame")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    fig, ax = plt.subplots()
    sns.lineplot(x="timestamp", y="PredictedYield", data=df, ax=ax, marker="o")
    ax.set_title("📈 Predicted Yield Trend Over Time")
    ax.set_xlabel("📅 Date")
    ax.set_ylabel("🌾 Yield (tons/ha)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

# 🔥 Run all functionalities
if __name__ == "__main__":
    create_database()   # Initialize database
    generate_map()      # Generate Folium map
    update_weather()    # Update weather data
    print("🚀 Visualizations and data updates completed successfully!")
