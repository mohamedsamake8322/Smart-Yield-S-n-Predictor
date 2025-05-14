import joblib
import numpy as np
import pandas as pd

def load_model():
    return joblib.load("yield_model.pkl")

def predict_single(model, temperature, humidity, precipitation, ph, fertilizer):
    X = np.array([[temperature, humidity, precipitation, ph, fertilizer]])
    return round(model.predict(X)[0], 2)

def predict_batch(model, df):
    features = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]
    df["Predicted_Yield"] = model.predict(df[features])
    return df

def suggest_improvements(yield_value, ph, fertilizer):
    suggestions = []
    if ph < 5.5:
        suggestions.append("Consider liming to increase soil pH.")
    elif ph > 7.5:
        suggestions.append("Consider adding sulfur to lower soil pH.")
    if fertilizer < 50:
        suggestions.append("You may need to increase fertilizer usage.")
    elif fertilizer > 300:
        suggestions.append("High fertilizer may not be cost-effective.")
    if not suggestions:
        return "No major improvements suggested. Maintain current conditions."
    return " ".join(suggestions)
model = joblib.load('yield_model.pkl')

def predict_yield(data: dict):
    features = np.array([[data['rainfall'], data['temperature'], data['humidity'], data['soil_type'], data['crop_type']]])
    return model.predict(features)[0]