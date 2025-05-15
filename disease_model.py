import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

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
        # Recréer l’architecture
        model = models.resnet18(pretrained=False)
        num_classes = 10  # Ajuste selon ton dataset
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Charger les poids
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
        
        return str(predicted_class_idx)  # Retourne l'indice de la classe prédite
    except Exception as e:
        raise ValueError(f"🧪 Prediction failed: {repr(e)}")