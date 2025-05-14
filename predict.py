import joblib
import pandas as pd

# Load the trained model
model = joblib.load('yield_model.pkl')

# Example input (replace with real input)
data = pd.DataFrame([{
    "soil_type": 2,
    "crop_type": 1,
    "temperature": 25.0,
    "rainfall": 100.0,
    "humidity": 60.0
}])

# Predict
prediction = model.predict(data)
print(f"🌾 Predicted yield: {prediction[0]:.2f}")
