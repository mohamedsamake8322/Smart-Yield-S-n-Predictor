import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# 📌 Dictionnaire des classes de maladies
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

# 📌 Définition du périphérique (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Définition du modèle
num_classes = len(CLASS_LABELS)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# 📌 Vérifier et créer le dossier `model/`
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 📦 Sauvegarde du modèle
model_path = os.path.join(model_dir, "disease_model.pth")
torch.save({"state_dict": model.state_dict()}, model_path)
print(f"✅ Modèle sauvegardé avec succès dans {model_path} !")

# 📌 Fonction de prédiction des maladies
def predict_disease(image_path):
    """Prédit la maladie des plantes à partir d'une image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        disease_name = CLASS_LABELS[predicted.item()]
    
    return disease_name

