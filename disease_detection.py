import time
from PIL import Image
from utils import predict_disease
from disease_info import get_disease_info, DISEASE_DATABASE
from disease_model import load_disease_model  # ✅ Added import

# ✅ Load the model BEFORE using it
start_time = time.time()
disease_model = load_disease_model("C:/Mohamed/model/disease_model.pth")  
end_time = time.time()

if disease_model:
    print("✅ Disease detection model is loaded successfully!")
else:
    print("🛑 Error: Disease model not loaded.")

print(f"⏳ Model loaded in {end_time - start_time:.2f} seconds.")

def process_image(image_file):
    """Converts an image to RGB format."""
    return Image.open(image_file).convert("RGB")

def detect_disease(disease_model=None, image=None, symptom=None):
    """Detects disease based on image or symptom."""
    if image and disease_model:
        label = predict_disease(disease_model, image)
        detected_plant = label.split()[0] if label else "Unknown"
        disease_details = get_disease_info(label) if label and label in DISEASE_DATABASE else None
    elif symptom:
        disease_details = next((d for d in DISEASE_DATABASE.values() if symptom.lower() in d.symptoms.lower()), None)
        label = disease_details.name if disease_details else "Unknown"
        detected_plant = "Unknown"
    else:
        return {"error": "Provide either an image or a symptom for detection."}

    return {
        "label": label,
        "plant": detected_plant,
        "info": disease_details or "⚠️ No matching disease found."
    }

# ✅ Example of symptom-based detection
symptom_query = "Water-soaked areas on leaves"
detected_disease = detect_disease(symptom=symptom_query)

if detected_disease.get("info"):  # ✅ Correction to avoid error
    print(f"Possible disease detected: {detected_disease['info']}")
else:
    print("No matching disease found.")

# ✅ Symptom-based detection only
def detect_disease_by_symptom(symptom):
    """🔎 Search for a disease by symptom."""
    return next((disease for disease in DISEASE_DATABASE.values() if symptom.lower() in disease.symptoms.lower()), None)

# ✅ Example usage
symptom_query = "Young seedlings develop rot at the crown"
detected_disease = detect_disease_by_symptom(symptom_query)

if detected_disease:
    print(f"Possible disease detected: {detected_disease.name}")  
else:
    print("No matching disease found.")

# ✅ Detection from the database
def detect_disease_from_database(symptom):
    """
    🔍 Detects a disease based on a symptom.
    - Searches in the `DISEASE_DATABASE`.
    - Returns the corresponding disease if found.
    """
    return next(
        (disease for disease in DISEASE_DATABASE.values() if symptom.lower() in disease.symptoms.lower()), 
        None
    )

# ✅ Example usage
symptom_query = "Soft, water-soaked lesions develop without discoloration"
detected_disease = detect_disease_from_database(symptom_query)

if detected_disease:
    print(f"Possible disease detected:\n{detected_disease}")
else:
    print("⚠️ No matching disease found.")
