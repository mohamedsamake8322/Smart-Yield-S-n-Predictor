# disease_model.py

import os
import joblib
import numpy as np
from PIL import Image
from typing import Union

def load_disease_model(model_path: str):
    """
    Load the trained disease model from the given path.

    Args:
        model_path (str): Path to the .pth model file.

    Returns:
        model: Trained model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: If loading fails for other reasons.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚫 Model file not found at: {model_path}")
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"❌ Error loading model: {str(e)}")


def preprocess_image(image: Image.Image, size: tuple = (128, 128)) -> np.ndarray:
    """
    Preprocess the input image to match the model input.

    Args:
        image (PIL.Image.Image): Input image.
        size (tuple): Resize target.

    Returns:
        np.ndarray: Flattened image array shaped for prediction.
    """
    image = image.convert("RGB").resize(size)
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize
    return img_array.flatten().reshape(1, -1)


def predict_disease(model, image: Union[Image.Image, np.ndarray]) -> str:
    """
    Predict the plant disease from an input image using the provided model.

    Args:
        model: Trained machine learning model.
        image (PIL.Image.Image): Input image.

    Returns:
        str: Predicted class label.
    """
    try:
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        return prediction[0]
    except Exception as e:
        raise ValueError(f"🧪 Prediction failed: {e}")
