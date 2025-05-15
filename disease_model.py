# disease_model.py

import joblib
import numpy as np
from PIL import Image
import os

def load_disease_model(model_path: str):
    """
    Load the trained disease model from the given path.

    Args:
        model_path (str): Path to the .pkl model file.

    Returns:
        Trained model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: For other loading errors.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def predict_disease(model, image: Image.Image):
    """
    Predict the plant disease from an input image using the provided model.

    Args:
        model: Trained machine learning model.
        image (PIL.Image.Image): Input image.

    Returns:
        str: Predicted class label.
    """
    # Resize and flatten image to match model input format
    img_array = np.array(image.resize((128, 128))).flatten().reshape(1, -1)
    
    prediction = model.predict(img_array)
    return prediction[0]
