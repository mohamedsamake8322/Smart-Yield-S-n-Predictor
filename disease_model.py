import os
import torch
import numpy as np
from PIL import Image
from typing import Union

def load_disease_model(model_path: str):
    """
    Load the trained disease model from the given path.

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
        model = torch.load(model_path, map_location=torch.device('cpu'))  # Assure le chargement sur CPU
        model.eval()  # Passe en mode évaluation
        return model
    except Exception as e:
        raise Exception(f"❌ Error loading model: {repr(e)}")


def preprocess_image(image: Image.Image, size: tuple = (128, 128)) -> torch.Tensor:
    """
    Preprocess the input image to match the model input.

    Args:
        image (PIL.Image.Image): Input image.
        size (tuple): Resize target.

    Returns:
        torch.Tensor: Image tensor ready for prediction.
    """
    image = image.convert("RGB").resize(size)
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)  # Change dimensions
    return img_tensor


def predict_disease(model, image: Union[Image.Image, np.ndarray]) -> str:
    """
    Predict the plant disease from an input image using the provided PyTorch model.

    Args:
        model: Trained PyTorch model.
        image (PIL.Image.Image): Input image.

    Returns:
        str: Predicted class label.
    """
    try:
        processed = preprocess_image(image)
        with torch.no_grad():  # Désactive le calcul du gradient pour la prédiction
            prediction = model(processed)
            predicted_class = prediction.argmax(dim=1).item()  # Récupère l'indice de la classe prédite
        return str(predicted_class)
    except Exception as e:
        raise ValueError(f"🧪 Prediction failed: {repr(e)}")