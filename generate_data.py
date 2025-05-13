import pandas as pd
import numpy as np
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Possible categories
soil_types = ['Sandy', 'Loamy', 'Clay', 'Silty']
crop_types = ['Maize', 'Rice', 'Wheat', 'Millet', 'Sorghum']

# Generate 1000 synthetic data points
def generate_data(n=1000):
    data = {
        'temperature': np.random.normal(loc=28, scale=5, size=n).round(2),  # °C
        'humidity': np.random.uniform(30, 90, n).round(2),  # %
        'pH': np.random.uniform(4.5, 8.5, n).round(2),
        'rainfall': np.random.gamma(shape=2, scale=30, size=n).round(1),  # mm
        'soil_type': np.random.choice(soil_types, n),
        'crop_type': np.random.choice(crop_types, n),
    }

    df = pd.DataFrame(data)

    # Generate yield based on some rules + noise
    def estimate_yield(row):
        base_yield = {
            'Maize': 2000,
            'Rice': 2500,
            'Wheat': 2200,
            'Millet': 1800,
            'Sorghum': 1900
        }[row['crop_type']]

        modifiers = 0
        if 6.0 <= row['pH'] <= 7.5:
            modifiers += 100
        if row['rainfall'] < 50:
            modifiers -= 300
        elif row['rainfall'] > 150:
            modifiers -= 200
        if row['temperature'] > 35:
            modifiers -= 150

        noise = np.random.normal(0, 100)
        return round(base_yield + modifiers + noise, 1)

    df['yield'] = df.apply(estimate_yield, axis=1)
    return df

# Generate and save to CSV
df = generate_data()
df.to_csv('data.csv', index=False)
print("✅ Data generated and saved to data.csv (1000 rows)")
