import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import random

# Dictionnaire des classes de maladies
CLASS_LABELS = {
    0: "Cucumber Anthracnose",
    1: "Eggplant Cercospora",
    2: "Eggplant Bacterial Wilt",
    3: "Eggplant Phomopsis",
    4: "Eggplant Healthy",
    5: "Okra Yellow Vein Mosaic Virus",
    6: "Okra Leaf Spot",
    7: "Okra Powdery Mildew",
    8: "Okra Caterpillar Damage",
    9: "Okra Healthy",
    10: "Okra Rust",
    11: "Okra Salinisation Stress",
    12: "Tomato Bacterial Spot",
    13: "Tomato Early Blight",
    14: "Tomato Healthy",
    15: "Tomato Late Blight",
    16: "Tomato Leaf Mold",
    17: "Tomato Septoria Leaf Spot",
    18: "Tomato Spider Mites",
    19: "Tomato Target Spot",
    20: "Tomato Yellow Leaf Curl Virus",
    21: "Maize Dwarf Mosaic Virus",
    22: "Barley Yellow Dwarf Virus",
    23: "Soybean Mosaic Virus",
    24: "Blossom-End Rot",
    25: "Chemical Damage",
    26: "Chimera Genetic Mutation",
    27: "Cracking Environmental Stress"
}

# Définition du périphérique (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle ResNet18 entraîné
def load_disease_model(model_path: str):
    """Charge le modèle de reconnaissance des maladies des plantes."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚫 Model file not found at: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)

        if 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]
        else:
            num_classes = len(CLASS_LABELS)

        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict, strict=True)  # Activation du mode strict pour éviter les incohérences
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"❌ Error loading model: {repr(e)}")

# Prétraitement des images avec une normalisation ajustable
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Prépare l’image pour l’analyse par le modèle de reconnaissance."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalisation plus neutre
    ])
    return transform(image).unsqueeze(0).to(device)

# Prédiction de la maladie à partir d’une image
def predict_disease(model, image: Image.Image) -> str:
    """Prédit la maladie végétale à partir de l’image."""
    try:
        processed = preprocess_image(image)
        with torch.no_grad():
            output = model(processed)
            predicted_class_idx = output.argmax(dim=1).item()
        return CLASS_LABELS.get(predicted_class_idx, f"Unknown ({predicted_class_idx})")
    except Exception as e:
        raise ValueError(f"🧪 Prediction failed: {repr(e)}")

# Modèle prédictif avancé basé sur les conditions climatiques
class DiseaseRiskPredictor:
    def __init__(self, disease_name, temperature, humidity, wind_speed, soil_type, insect_population, crop_stage, season):
        self.disease_name = disease_name.lower()
        self.temperature = temperature
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.soil_type = soil_type
        self.insect_population = insect_population
        self.crop_stage = crop_stage
        self.season = season.lower()

    def get_seasonal_adjustment(self):
        """Ajuste le risque en fonction de la saison."""
        season_factors = {
            "printemps": 0.1, "été": 0.2, "automne": 0.15, "hiver": 0.05
        }
        return season_factors.get(self.season, 0)

    def calculate_risk(self):
        """Calcule le risque d'infection basé sur les conditions environnementales."""
        disease_factors = {
            "viral": {"temperature": [25, 35], "humidity": [50, 80], "vector": "aphid"},
            "bacterial": {"temperature": [18, 30], "humidity": [70, 100], "vector": None},
            "fungal": {"temperature": [10, 25], "humidity": [80, 100], "vector": None},
            "phytoplasma": {"temperature": [20, 32], "humidity": [60, 90], "vector": "leafhopper"},
            "abiotic": {"temperature": [22, 38], "humidity": [40, 70], "vector": None}
        }

        base_risk = 0.5  # Base neutre plutôt qu'un nombre aléatoire

        if self.disease_name in disease_factors:
            factors = disease_factors[self.disease_name]
            if factors["temperature"][0] <= self.temperature <= factors["temperature"][1]:
                base_risk += 0.15
            if factors["humidity"][0] <= self.humidity <= factors["humidity"][1]:
                base_risk += 0.20
            if self.wind_speed > 20 and self.disease_name == "fungal":
                base_risk += 0.25
            if self.soil_type == "argileux" and self.disease_name == "bacterial":
                base_risk += 0.10
            if factors["vector"] and self.insect_population > 500:
                base_risk += 0.30

        base_risk += self.get_seasonal_adjustment()
        base_risk = min(base_risk, 1)
        return f"🔍 Risque estimé d'infection pour {self.disease_name}: {base_risk:.2f} (0 = faible, 1 = élevé)"

# Exemple d'utilisation
disease_name = "viral"
predictor = DiseaseRiskPredictor(disease_name, 28, 65, 10, "sableux", 600, "jeunes plants", "été")
print(predictor.calculate_risk())
print("Exécution terminée avec succès !")
# Enregistrer le modèle entraîné
model_path = "model/disease_model.pth"
torch.save({"state_dict": model.state_dict()}, model_path)
print(f"✅ Modèle sauvegardé avec succès dans {model_path} !")
