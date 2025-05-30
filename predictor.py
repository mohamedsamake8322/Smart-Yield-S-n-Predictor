import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

# ✅ Définition du périphérique (CPU uniquement)
device = torch.device("cpu")

# ---------- Détection automatique des colonnes ----------
def detect_input_size(csv_path="data.csv"):
    """Détecte automatiquement le nombre de colonnes de features du CSV."""
    df = pd.read_csv(csv_path)
    input_size = len(df.columns) - 1  # 🚀 Ignorer la colonne cible (ex: 'Yield')
    print(f"✅ Détection des features : {input_size} colonnes utilisées pour la prédiction.")
    return input_size

# ---------- Définition du modèle PyTorch ----------
class PyTorchModel(nn.Module):
    def __init__(self, input_size):
        super(PyTorchModel, self).__init__()

        # ✅ Vérification et conversion de input_size en entier
        if not isinstance(input_size, int):
            try:
                input_size = int(input_size)
            except ValueError:
                raise TypeError(f"🛑 input_size must be an integer, but got {type(input_size)}")

        self.fc1 = nn.Linear(input_size, 64).to(device)
        self.fc2 = nn.Linear(64, 32).to(device)
        self.fc3 = nn.Linear(32, 1).to(device)

    def forward(self, x):
        x = x.to(device)  # 🚀 Envoi des données vers CPU
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------- Fertilizer Mapping ----------
fertilizer_map = {
    "DAP": 70,
    "Urea": 80,
    "Compost": 50,
    "NPK": 65,
    "None": 0
}

def preprocess_fertilizer_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Fertilizer" in df.columns:
        df["Fertilizer"] = df["Fertilizer"].apply(lambda val: fertilizer_map.get(val, None) if isinstance(val, str) else val)
    return df

# ---------- Model Persistence ----------
MODEL_PATH = "model/disease_model.pth"

def load_model(input_size, path=MODEL_PATH):
    """Charge le modèle PyTorch et ajuste automatiquement le nombre de features."""
    if not isinstance(input_size, int):
        try:
            input_size = int(input_size)
        except ValueError:
            raise TypeError(f"🛑 input_size must be an integer, but got {type(input_size)}")

    model = PyTorchModel(input_size)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))  # ✅ Charge sur CPU
        model.eval()  
        print(f"✅ Modèle PyTorch chargé avec succès depuis {path} !")
    else:
        print(f"🚫 Modèle non trouvé à {path}.")
    return model

def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Modèle PyTorch sauvegardé sous {path}.")

# ---------- Nettoyage automatique du CSV ----------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données pour éviter les erreurs avec PyTorch."""
    print("🔄 Vérification et nettoyage du dataset...")
    df = df.apply(pd.to_numeric, errors="coerce")  # 🚀 Convertit toutes les valeurs en numériques
    df.fillna(0, inplace=True)  # ✅ Remplace les valeurs NaN par 0
    print("✅ Nettoyage terminé.")
    return df

# ---------- Single Prediction ----------
def predict_single(model, features: dict):
    """Effectue une prédiction unique."""
    input_df = pd.DataFrame([features])
    input_df = preprocess_fertilizer_column(input_df)
    input_df["NDVI"] = 0.5

    input_tensor = torch.tensor(input_df.values, dtype=torch.float32).to(device)  # ✅ Ajout de `.to(device)`
    prediction = model(input_tensor).item()
    return prediction

# ---------- Batch Prediction ----------
def predict_batch(model, df: pd.DataFrame):
    """Effectue des prédictions sur plusieurs données."""
    df = preprocess_fertilizer_column(df)
    df["NDVI"] = 0.5

    df = clean_dataframe(df)  # 🚀 Nettoyage automatique avant conversion

    required_features = list(df.columns)  # ✅ Utiliser les colonnes dynamiques
    input_tensor = torch.tensor(df[required_features].values, dtype=torch.float32).to(device)
    
    predictions = model(input_tensor).detach().numpy()
    return predictions

# ---------- Exécution autonome du script ----------
if __name__ == "__main__":
    print("🔄 Détection automatique des features...")
    input_size = detect_input_size()

    print("🔄 Chargement du modèle...")
    model = load_model(input_size=input_size)

    # 🔥 Test rapide de prédiction avec des valeurs fictives
    example_features = {
        "Temperature": 25,
        "Humidity": 60,
        "Precipitation": 12,
        "pH": 6.5,
        "Fertilizer": 80,
        "NDVI": 0.5
    }
    prediction = predict_single(model, example_features)
    print(f"🌾 Prédiction du rendement: {prediction:.2f} tonnes/hectare")
