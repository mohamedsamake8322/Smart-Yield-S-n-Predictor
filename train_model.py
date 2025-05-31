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
from sklearn.preprocessing import StandardScaler

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


# 📥 Chargement et prétraitement des données
def load_data(df):
    logging.info("🔄 Prétraitement du dataset...")

    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    
    X = df.drop(columns=["yield"])
    y = df["yield"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

# 🔥 Définition du modèle PyTorch
class PyTorchModel(nn.Module):
    def __init__(self, input_size):
        super(PyTorchModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 64).to(device)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32).to(device)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1).to(device)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.to(device)
        x = self.activation(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 📌 Sauvegarde et chargement du modèle
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pth")

def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
    logging.info(f"✅ Modèle PyTorch sauvegardé sous {path}.")

# 🚀 Fonction pour entraîner le modèle
def train_model():
    logging.info("🚀 Début de l'entraînement du modèle...")

    input_size, df = detect_input_size()
    X_train, X_test, y_train, y_test = load_data(df)

    model = PyTorchModel(input_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

    for epoch in range(1000):
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    save_model(model)

    return model

# 🚀 Fin de l'entraînement
logging.info("🎯 ✅ Modèle prêt à être utilisé !")
