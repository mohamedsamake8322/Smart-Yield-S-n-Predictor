import torch
import torch.nn as nn
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler

# ✅ Définition du périphérique (CPU uniquement)
device = torch.device("cpu")

# ✅ Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔎 Détection automatique des colonnes
def detect_input_size(csv_path="data.csv"):
    df = pd.read_csv(csv_path)
    logging.info(f"🔎 Colonnes disponibles dans le dataset : {df.columns.tolist()}")

    if "yield" not in df.columns:
        raise KeyError("🛑 Erreur : La colonne 'yield' n'existe pas dans le dataset. Vérifie ton fichier CSV.")

    input_size = len(df.columns) - 1
    return int(input_size), df  # ✅ Convertir en entier pour éviter l’erreur

# 🔎 Nettoyage et Normalisation
def clean_and_normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("🔄 Vérification et normalisation du dataset...")

    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])  
    logging.info("✅ Normalisation terminée.")

    return df

# 🔎 Conversion des valeurs catégoriques
def convert_categorical_features(features):
    """Convertir `soil_type` et `crop_type` en valeurs numériques."""
    conversions = {
        "soil_type": {"sandy": 1, "clay": 0},
        "crop_type": {"wheat": 1, "corn": 0}
    }
    for feature in conversions:
        if feature in features:
            features[feature] = conversions[feature].get(features[feature], -1)
    return features

# ✅ Définition du modèle PyTorch
class PyTorchModel(nn.Module):
    def __init__(self, input_size):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64).to(device)
        self.fc2 = nn.Linear(64, 32).to(device)
        self.fc3 = nn.Linear(32, 1).to(device)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.to(device)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 🔎 Sauvegarde et Chargement du modèle
MODEL_PATH = "model/disease_model.pth"

def save_model(model, path=MODEL_PATH):
    """Sauvegarde correcte du modèle."""
    torch.save(model.state_dict(), path)
    logging.info(f"✅ Modèle PyTorch sauvegardé sous {path}.")

def load_model(input_size, path=MODEL_PATH):
    """Charge le modèle PyTorch et vérifie la compatibilité."""
    model = PyTorchModel(input_size)

    if not os.path.exists(path):
        logging.error(f"🚫 Modèle non trouvé à {path}. Vérifie l'entraînement.")
        raise FileNotFoundError(f"Modèle non trouvé : {path}")

    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        logging.info(f"✅ Modèle chargé avec succès depuis {path} !")
    except RuntimeError as e:
        logging.error(f"🛑 Erreur de chargement du modèle : {e}")
        raise RuntimeError("Le fichier du modèle n'est pas compatible avec l'architecture actuelle.")
    
    return model

# 🔎 Prédiction
def predict_single(model, features: dict):
    """Effectue une prédiction unique."""
    features = convert_categorical_features(features)  # ✅ Correction ajoutée
    input_df = pd.DataFrame([features])
    input_df = clean_and_normalize_dataframe(input_df)

    input_tensor = torch.tensor(input_df.values, dtype=torch.float32).to(device)
    return model(input_tensor).item()

# 🔥 Test rapide
if __name__ == "__main__":
    logging.info("🔄 Détection automatique des features...")
    input_size, df = detect_input_size()

    model = load_model(input_size=input_size)

    example_features = {
        "temperature": 25,
        "humidity": 60,
        "pH": 6.5,
        "rainfall": 12,
        "soil_type": "sandy",
        "crop_type": "wheat"
    }

    example_features = convert_categorical_features(example_features)
    prediction = predict_single(model, example_features)
    logging.info(f"🌾 Prédiction du rendement: {prediction:.2f} tonnes/hectare")
    logging.info("🎯 Script terminé avec succès !")
