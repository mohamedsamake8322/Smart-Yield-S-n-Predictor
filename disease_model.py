import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# Dictionnaire des classes (index -> nom de la classe)
CLASS_LABELS = {
    0: "Cucumber Anthracnose",
    1: "Eggplant Cercospora",
    2: "Eggplant___Bacterial_wilt",
    3: "Eggplant___Phomopsis",
    4: "Eggplant_Healthly",
    5: "Okra____Yellow_Vein_Mosaic_Virus",
    6: "Okra___Leaf_spot",
    7: "Okra__Powdery_mildew",
    8: "Okra_catepillar",
    9: "Okra_Healthly",
    10: "Okra_rust",
    11: "Okra_salinisation",
    12: "Tomato___Bacterial spot",
    13: "Tomato___Early_blight",
    14: "Tomato___Healthy",
    15: "Tomato___Late_blight",
    16: "Tomato___Leaf_Mold",
    17: "Tomato___Septoria_leaf spot",
    18: "Tomato___Spider_mites",
    19: "Tomato___Target spot",
    20: "Tomato___Yellow_Leaf_Curl_Virus"
}

def load_disease_model(model_path: str):
    """
    Load the trained ResNet18 model for plant disease detection.

    Args:
        model_path (str): Path to the .pth model file.

    Returns:
        model: Trained PyTorch model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: If loading fails for other reasons.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚫 Model file not found at: {model_path}")

    try:
        # Déterminer dynamiquement le nombre de classes
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('state_dict', checkpoint)
        fc_weight = state_dict.get('fc.weight')
        num_classes = fc_weight.shape[0] if fc_weight is not None else 21

        # Recréer l’architecture
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Charger les poids
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"❌ Error loading model: {repr(e)}")

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess the input image for the PyTorch model.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Taille adaptée à ResNet18
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalisation ImageNet
    ])
    
    return transform(image).unsqueeze(0)  # Ajoute une dimension batch

def predict_disease(model, image: Image.Image) -> str:
    """
    Predict the plant disease using the trained ResNet18 model.

    Args:
        model: Trained PyTorch model.
        image (PIL.Image.Image): Input image.

    Returns:
        str: Predicted class label.
    """
    try:
        processed = preprocess_image(image)

        # Prédiction
        with torch.no_grad():
            output = model(processed)
            predicted_class_idx = output.argmax(dim=1).item()
        
        class_label = CLASS_LABELS.get(predicted_class_idx, f"Unknown ({predicted_class_idx})")
        return class_label
    except Exception as e:
        raise ValueError(f"🧪 Prediction failed: {repr(e)}")
