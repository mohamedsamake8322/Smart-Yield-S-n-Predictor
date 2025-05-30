import os
import joblib
import torch
import torchvision
import pandas as pd

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

# ---------- Paths ----------
MODEL_PATH = "model/model_xgb.pkl"
DISEASE_MODEL_PATH = "model/disease_model.pth"

# ---------- Disease Class Mapping ----------
disease_class_names = {
    0: "Tâche bactérienne",
    1: "Tâche brune",
    2: "Curl Virus",
    3: "Mildiou",
    4: "Mosaïque",
    5: "Rouille",
    6: "Tâche jaune",
    7: "Feuille saine",
    8: "Septoriose",
    9: "Autre"
}

# ---------- PyTorch Model Definition ----------
class MyCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ---------- Model Persistence ----------
def load_model(path: str = MODEL_PATH):
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

# ---------- Disease Model Loading ----------
def load_disease_model():
    if not os.path.exists(DISEASE_MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file {DISEASE_MODEL_PATH} not found.")

    data = torch.load(DISEASE_MODEL_PATH, map_location=torch.device("cpu"))

    if isinstance(data, dict) and "state_dict" in data:
        model = MyCNN()
        model.load_state_dict(data["state_dict"])
        model.eval()
        print("✅ Disease model loaded successfully!")
        return model
    else:
        raise ValueError("🚫 Model state dict not found in the checkpoint.")

# 📌 Load models at startup
disease_model = load_disease_model()
crop_model = load_model()

# ---------- Disease Prediction ----------
def predict_disease(model, image_file):
    if model is None:
        raise ValueError("🚫 Aucun modèle de détection de maladie chargé.")

    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))

    if isinstance(model, torch.nn.Module):
        transform = torchvision.transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(transform)
            predicted_class = outputs.argmax(dim=1).item()
            return disease_class_names.get(predicted_class, "Inconnu")
    return "⚠️ Modèle non compatible."

# ---------- Image Preprocessing (if needed externally) ----------
def process_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))
    return image

# ---------- Crop Prediction ----------
def predict_single(model, features: dict):
    input_df = pd.DataFrame([features])
    input_df = preprocess_fertilizer_column(input_df)
    input_df["NDVI"] = 0.5
    return model.predict(input_df)[0]

def predict_batch(model, df: pd.DataFrame):
    df = preprocess_fertilizer_column(df)
    df["NDVI"] = 0.5

    required_features = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(f"🚫 Missing columns: {missing}")

    return model.predict(df[required_features])

# ---------- Training ----------
def train_model(df: pd.DataFrame, model_type="RandomForest"):
    df["NDVI"] = 0.5
    df = preprocess_fertilizer_column(df)

    X = df[["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]]
    y = df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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