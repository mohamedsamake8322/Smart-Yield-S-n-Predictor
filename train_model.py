import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Définition du périphérique (CPU uniquement)
device = torch.device("cpu")

# ✅ Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📂 Vérification et création du dossier "model"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
logging.info(f"✅ Directory verified: {MODEL_DIR}")

# 📥 Détection automatique des colonnes du dataset
def detect_input_size(csv_path="data.csv"):
    """Détecte automatiquement le nombre de colonnes de features du CSV."""
    df = pd.read_csv(csv_path)
    logging.info(f"🔎 Colonnes disponibles dans le dataset : {df.columns.tolist()}")

    if "yield" not in df.columns:
        raise KeyError("🛑 Erreur : La colonne 'yield' n'existe pas dans le dataset. Vérifie ton fichier CSV.")

    input_size = len(df.columns) - 1  # 🚀 Ignorer la colonne cible (ex: 'yield')
    logging.info(f"✅ Détection des features : {input_size} colonnes utilisées pour l'entraînement.")
    return input_size, df

# 📥 Chargement et prétraitement des données
def load_data(df):
    logging.info("🔄 Prétraitement du dataset...")
    
    # ✅ Nettoyage des données
    df = df.apply(pd.to_numeric, errors="coerce")  # 🔥 Convertir toutes les valeurs en numériques
    df.fillna(0, inplace=True)  # ✅ Remplacer les NaN par 0

    # ✅ Séparation des features et de la cible
    X = df.drop(columns=["yield"])
    y = df["yield"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Détection du input_size et chargement des données
try:
    input_size, df = detect_input_size()
    X_train, X_test, y_train, y_test = load_data(df)
except KeyError as e:
    logging.error(str(e))
    exit(1)

# 🔥 Définition du modèle PyTorch
class PyTorchModel(nn.Module):
    def __init__(self, input_size):
        super(PyTorchModel, self).__init__()

        # ✅ Adaptabilité automatique au nombre d'entrées
        self.fc1 = nn.Linear(input_size, 64).to(device)
        self.fc2 = nn.Linear(64, 32).to(device)
        self.fc3 = nn.Linear(32, 1).to(device)

    def forward(self, x):
        x = x.to(device)  # ✅ Envoi des données vers CPU
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 📌 Instanciation du modèle
model = PyTorchModel(input_size)

# 🔧 Définition de la fonction de coût et de l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ✅ Conversion des données en tensors avec passage sur CPU
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

# 🚀 Entraînement du modèle PyTorch
for epoch in range(500):
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 📊 Évaluation du modèle
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

with torch.no_grad():
    predictions = model(X_test_tensor)
    rmse = mean_squared_error(y_test_tensor.numpy(), predictions.numpy(), squared=False)
    r2 = r2_score(y_test_tensor.numpy(), predictions.numpy())

metrics = {
    "rmse": float(rmse),
    "r2": float(r2)
}

# 💾 Sauvegarde du modèle
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pth")
torch.save(model.state_dict(), MODEL_PATH)  # ✅ Sauvegarde correcte
logging.info(f"✅ Model saved successfully in {MODEL_PATH}")

# 📊 Sauvegarde des métriques
METRICS_PATH = os.path.join(MODEL_DIR, "retrained_model_metrics.json")
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)
logging.info(f"📊 Metrics logged in {METRICS_PATH}")

# 🚀 Fin de l'entraînement
logging.info("🎯 ✅ Modèle entraîné et prêt à être utilisé !")
