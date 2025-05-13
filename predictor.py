# === predictor.py ===
import numpy as np
import joblib

MODEL_PATH = "models/yield_model.pkl"

# Load trained model
def load_model():
    return joblib.load(MODEL_PATH)

# Predict single entry
def predict_single(model, temperature, humidity, precipitation, ph, fertilizer):
    features = np.array([[temperature, humidity, precipitation, ph, fertilizer]])
    prediction = model.predict(features)
    return round(prediction[0], 2)

# Predict batch from dataframe
def predict_batch(model, df):
    features = df[["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]]
    df["Predicted_Yield"] = model.predict(features).round(2)
    return df

# Generate recommendation based on yield and input values
def get_recommendation(predicted_yield, ph, fertilizer):
    if predicted_yield < 20:
        if ph < 5.5:
            return "🔎 pH too acidic. Add limestone."
        elif ph > 8:
            return "⚠️ pH too basic. Check soil alkalinity."
        elif fertilizer < 50:
            return "💡 Low fertilization. Increase the fertilizer dose."
        else:
            return "🌱 Suboptimal conditions. Monitor humidity and temperature."
    elif predicted_yield >= 30:
        return "✅ Excellent performance expected. Keep it up !"
    else:
        return "ℹ️ Average yield. Gradually optimize the settings.."
