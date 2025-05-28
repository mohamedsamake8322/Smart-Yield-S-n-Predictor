import pandas as pd
import numpy as np

# Définir des valeurs fictives
data = {
    "crop": ["Tomato", "Wheat", "Corn", "Rice", "Pepper"],
    "pH": np.random.uniform(5.5, 7.5, 5),
    "soil_type": ["Clay", "Sandy", "Loamy", "Clay", "Sandy"],
    "growth_stage": ["Seedling", "Vegetative", "Flowering", "Maturity", "Harvest"],
    "temperature": np.random.uniform(15, 35, 5),
    "humidity": np.random.uniform(40, 80, 5),
    "recommended_fertilizer": ["NPK", "Urea", "Compost", "NPK", "Organic"]
}

# Convertir en DataFrame
df = pd.DataFrame(data)

# Sauvegarder en CSV
df.to_csv("fertilization_data.csv", index=False)

print("fertilization_data.csv créé avec succès !")
