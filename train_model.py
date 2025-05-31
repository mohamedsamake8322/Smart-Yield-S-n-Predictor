import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Définition du périphérique (CPU uniquement)
device = torch.device("cpu")

# ✅ Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📂 Vérification et création du dossier "model"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
logging.info(f"✅ Dossier modèle vérifié : {MODEL_DIR}")

# 📥 Détection automatique des colonnes du dataset
def detect_input_size(csv_path="data.csv"):
    df = pd.read_csv(csv_path)
    logging.info(f"🔎 Colonnes disponibles dans le dataset : {df.columns.tolist()}")

    if "yield" not in df.columns:
        raise KeyError("🛑 Erreur : La colonne 'yield' n'existe pas dans le dataset.")

    return len(df.columns) - 1, df

# 📥 Chargement et prétraitement des données
def load_data(df):
    logging.info("🔄 Prétraitement du dataset...")

    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(df.mean(), inplace=True)  # ✅ Remplacement des `NaN` par la moyenne

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
        self.activation = nn.LeakyReLU()
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

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"✅ Modèle PyTorch correctement sauvegardé sous {MODEL_PATH}.")

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

    if not os.path.exists(MODEL_PATH):
        logging.error("🛑 Erreur : `disease_model.pth` n’a pas été sauvegardé correctement.")
        raise RuntimeError("Le modèle n'a pas été sauvegardé.")

    return model

# 🚀 Lancement automatique de l'entraînement
if __name__ == "__main__":
    train_model()
