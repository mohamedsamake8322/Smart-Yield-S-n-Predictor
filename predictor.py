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

# ---------- Détection automatique des colonnes ----------
def detect_input_size(csv_path="data.csv"):
    df = pd.read_csv(csv_path)
    logging.info(f"🔎 Colonnes disponibles dans le dataset : {df.columns.tolist()}")

    if "yield" not in df.columns:
        raise KeyError("🛑 Erreur : La colonne 'yield' n'existe pas dans le dataset. Vérifie ton fichier CSV.")

    input_size = len(df.columns) - 1

    try:
        input_size = int(input_size)  # ✅ Convertir en entier pour éviter l’erreur
    except ValueError:
        logging.error(f"🛑 Erreur : `input_size` doit être un entier, mais reçu {type(input_size)}")
        raise TypeError(f"input_size must be an integer, but got {type(input_size)}")

    logging.info(f"✅ Détection des features : {input_size} colonnes utilisées pour la prédiction.")
    return input_size, df


# ---------- Définition du modèle PyTorch ----------
class PyTorchModel(nn.Module):
    def __init__(self, input_size):
        super(PyTorchModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 64).to(device)
        self.batch_norm1 = nn.BatchNorm1d(64)  # ✅ Ajout de BatchNorm
        self.fc2 = nn.Linear(64, 32).to(device)
        self.batch_norm2 = nn.BatchNorm1d(32)  # ✅ Ajout de BatchNorm
        self.fc3 = nn.Linear(32, 1).to(device)

        self.activation = nn.LeakyReLU(negative_slope=0.01)  # ✅ Activation améliorée
        self.dropout = nn.Dropout(0.3)  # ✅ Régularisation pour éviter l'overfitting

    def forward(self, x):
        x = x.to(device)
        x = self.activation(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
# ---------- Model Persistence ----------
MODEL_PATH = "model/disease_model.pth"

def load_model(input_size, path=MODEL_PATH):
    """Charge le modèle PyTorch et vérifie la compatibilité avec `input_size`."""
    model = PyTorchModel(input_size)
    
    if not os.path.exists(path):
        logging.error(f"🚫 Modèle non trouvé à {path}. Vérifie que l'entraînement a bien eu lieu.")
        raise FileNotFoundError(f"Modèle non trouvé : {path}")
    
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        logging.info(f"✅ Modèle PyTorch chargé avec succès depuis {path} !")
    except RuntimeError as e:
        logging.error(f"🛑 Erreur de chargement du modèle : {e}")
        raise RuntimeError("Le fichier du modèle n'est pas compatible avec l'architecture actuelle.")
    
    return model

def save_model(model, path=MODEL_PATH):
    """Sauvegarde le modèle PyTorch."""
    try:
        torch.save(model.state_dict(), path)
        logging.info(f"✅ Modèle PyTorch sauvegardé sous {path}.")
    except Exception as e:
        logging.error(f"🛑 Erreur lors de la sauvegarde du modèle : {e}")
        raise RuntimeError("Impossible de sauvegarder le modèle.")

# ---------- Nettoyage et Normalisation du CSV ----------
def clean_and_normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et normalise les données pour éviter les erreurs avec PyTorch."""
    logging.info("🔄 Vérification et normalisation du dataset...")

    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])  # ✅ Normalisation
    logging.info("✅ Normalisation terminée.")

    return df

# ---------- Single Prediction ----------
def predict_single(model, features: dict):
    """Effectue une prédiction unique."""
    input_df = pd.DataFrame([features])
    input_df = clean_and_normalize_dataframe(input_df)

    input_tensor = torch.tensor(input_df.values, dtype=torch.float32).to(device)
    prediction = model(input_tensor).item()
    return prediction

# ---------- Batch Prediction ----------
def predict_batch(model, df: pd.DataFrame):
    """Effectue des prédictions sur plusieurs données."""
    df = clean_and_normalize_dataframe(df)

    required_features = list(df.columns)
    input_tensor = torch.tensor(df[required_features].values, dtype=torch.float32).to(device)
    
    predictions = model(input_tensor).detach().numpy()
    return predictions

# ---------- Exécution autonome du script ----------
if __name__ == "__main__":
    logging.info("🔄 Détection automatique des features...")
    try:
        input_size, df = detect_input_size()
    except KeyError as e:
        logging.error(str(e))
        exit(1)

    # 🚨 Vérification finale du type de `input_size`
    if not isinstance(input_size, int):
        logging.error(f"🛑 `input_size` doit être un entier, mais reçu {type(input_size)} avec valeur `{input_size}`")
        raise TypeError(f"`input_size` must be an integer, but got {type(input_size)}")

    logging.info("🔄 Chargement du modèle...")
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
    logging.info(f"🌾 Prédiction du rendement: {prediction:.2f} tonnes/hectare")
