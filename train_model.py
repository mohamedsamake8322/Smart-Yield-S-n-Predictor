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

# ✅ Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📂 Vérification et création du dossier "model"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
logging.info(f"✅ Directory verified: {MODEL_DIR}")

# 📥 Chargement et prétraitement des données
DATA_PATH = "data.csv"

def load_data(path):
    if not os.path.exists(path):
        logging.error("❌ Dataset not found.")
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    logging.info("🔄 Loading dataset...")
    df = pd.read_csv(path)

    if "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["month"] = pd.to_datetime(df["date"]).dt.month

    df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])
    X = df_encoded.drop(columns=["yield"])
    y = df_encoded["yield"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data(DATA_PATH)

# 🔥 Définition du modèle PyTorch
class PyTorchModel(nn.Module):
    def __init__(self, input_size):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 📌 Instanciation du modèle
input_size = X_train.shape[1]
model = PyTorchModel(input_size)

# 🔧 Définition de la fonction de coût et de l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 🚀 Entraînement du modèle PyTorch
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

for epoch in range(500):
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 📊 Évaluation du modèle
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    predictions = model(X_test_tensor)
    rmse = mean_squared_error(y_test_tensor.numpy(), predictions.numpy(), squared=False)
    r2 = r2_score(y_test_tensor.numpy(), predictions.numpy())

metrics = {"rmse": rmse, "r2": r2}
logging.info(f"✅ Model trained. RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.2f}")

# 💾 Sauvegarde du modèle
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pth")
torch.save(model.state_dict(), MODEL_PATH)
logging.info(f"✅ Model saved successfully in {MODEL_PATH}")

# 📊 Sauvegarde des métriques
METRICS_PATH = os.path.join(MODEL_DIR, "retrained_model_metrics.json")
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)
logging.info(f"📊 Metrics logged in {METRICS_PATH}")
