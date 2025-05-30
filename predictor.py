import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

# ---------- Définition du modèle PyTorch ----------
class PyTorchModel(nn.Module):
    def __init__(self, input_size):
        super(PyTorchModel, self).__init__()

        # ✅ Vérification et conversion de input_size en entier
        if not isinstance(input_size, int):
            try:
                input_size = int(input_size)  # Force la conversion en entier si nécessaire
            except ValueError:
                raise TypeError(f"🛑 input_size must be an integer, but got {type(input_size)}")

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Couche de sortie (régresseur)

    def forward(self, x):
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
    # ✅ Vérification et conversion de input_size en entier avant d'initialiser le modèle
    if not isinstance(input_size, int):
        try:
            input_size = int(input_size)
        except ValueError:
            raise TypeError(f"🛑 input_size must be an integer, but got {type(input_size)}")

    model = PyTorchModel(input_size)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()  # Mode évaluation
        print(f"✅ Modèle PyTorch chargé avec succès depuis {path} !")
    else:
        print(f"🚫 Modèle non trouvé à {path}.")
    return model

def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Modèle PyTorch sauvegardé sous {path}.")

# ---------- Single Prediction ----------
def predict_single(model, features: dict):
    input_df = pd.DataFrame([features])
    input_df = preprocess_fertilizer_column(input_df)
    input_df["NDVI"] = 0.5  # Valeur NDVI par défaut

    # Convertir les données en tensor PyTorch
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
    prediction = model(input_tensor).item()  # Obtenir une seule valeur
    return prediction

# ---------- Batch Prediction ----------
def predict_batch(model, df: pd.DataFrame):
    df = preprocess_fertilizer_column(df)
    df["NDVI"] = 0.5  # Valeur NDVI par défaut
    required_features = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]

    # Convertir en tensor PyTorch
    input_tensor = torch.tensor(df[required_features].values, dtype=torch.float32)
    predictions = model(input_tensor).detach().numpy()  # Convertir en NumPy pour le retour
    return predictions

# ---------- Training ----------
def train_model(df: pd.DataFrame):
    df["NDVI"] = 0.5
    df = preprocess_fertilizer_column(df)

    X = df[["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]]
    y = df["Yield"]

    # ✅ Vérification de input_size avant utilisation
    input_size = X.shape[1]
    if not isinstance(input_size, int):
        raise TypeError(f"🛑 input_size should be an integer, but got {type(input_size)}")

    # Convertir les données en tensors PyTorch
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    model = PyTorchModel(input_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Boucle d'entraînement
    for epoch in range(500):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    print(f"[INFO] Modèle entraîné avec succès.")
    save_model(model)

    return model