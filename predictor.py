import joblib
import pandas as pd
import os
import torch
import torchvision

from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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
MODEL_PATH = "model/model_xgb.pkl"
DISEASE_MODEL_PATH = "model/disease_model.pth"

def load_model(path: str = MODEL_PATH):
    """Charge un modèle de prédiction des cultures en toute sécurité."""
    if not os.path.exists(path):
        print(f"[ERROR] Model file not found at {path}")
        return None

    try:
        model = joblib.load(path)
        print(f"[INFO] Model loaded successfully from {path}")
        return model
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None

def save_model(model, path: str = MODEL_PATH):
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}.")
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = torch.nn.Linear(224 * 224 * 3, 10)  # Exemple

    def forward(self, x):
        return self.fc(x)

# ---------- Disease Model Loading ----------
def load_disease_model():
    """Charge le modèle de détection des maladies basé sur PyTorch."""
    if not os.path.exists(DISEASE_MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file {DISEASE_MODEL_PATH} not found.")

    data = torch.load(DISEASE_MODEL_PATH, map_location=torch.device("cpu"))
    
    if isinstance(data, dict) and "model_state_dict" in data:
        model = MyModel()  # 🚀 Remplace `MyModel()` par ta classe PyTorch
        model.load_state_dict(data["model_state_dict"])
        model.eval()  # ✅ Mets le modèle en mode évaluation
        print("✅ Disease model loaded successfully!")
        return model
    else:
        raise ValueError("🚫 Model state dict not found in the checkpoint.")

# 📌 Chargement des modèles
disease_model = load_disease_model()
crop_model = load_model()

# ---------- Single Prediction ----------
def predict_single(model, features: dict):
    input_df = pd.DataFrame([features])
    input_df = preprocess_fertilizer_column(input_df)
    input_df["NDVI"] = 0.5
    return model.predict(input_df)[0]

# ---------- Batch Prediction ----------
def predict_batch(model, df: pd.DataFrame):
    df = preprocess_fertilizer_column(df)
    df["NDVI"] = 0.5

    required_features = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]
    missing_features = [col for col in required_features if col not in df.columns]

    if missing_features:
        raise ValueError(f"🚫 Missing columns in dataset: {missing_features}")

    return model.predict(df[required_features])

def train_model(df: pd.DataFrame, model_type="RandomForest"):
    """Entraîne un modèle basé sur le type sélectionné."""
    df["NDVI"] = 0.5
    df = preprocess_fertilizer_column(df)

    X = df[["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]]
    y = df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔹 Sélection du type de modèle
    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        import xgboost as xgb
        model = xgb.XGBRegressor(enable_categorical=True)
    else:
        raise ValueError(f"🚫 Model type {model_type} not recognized!")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[INFO] Model trained ({model_type}). MAE: {mae:.2f}")

    return model

# ---------- Disease Prediction ----------
def predict_disease(model, image_file):
    """Prédiction de la maladie à partir d'une image."""
    if model is None:
        raise ValueError("🚫 Aucun modèle de détection de maladie chargé.")

    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))

    if isinstance(model, torch.nn.Module):
        transform = torchvision.transforms.ToTensor()(image)
        transform = transform.unsqueeze(0)
        prediction = model(transform)
        return prediction.argmax().item()

    return "⚠️ Modèle non compatible."

# ---------- Image Processing ----------
def process_image(image_file):
    """Prétraitement de l'image pour la détection des maladies."""
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))
    return image
