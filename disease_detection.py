# disease_detection.py
from PIL import Image
from utils import predict_disease
from disease_info import get_disease_info, DISEASE_DATABASE
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
# ✅ Example usage based on symptoms:
symptom_query = "Water-soaked areas on leaves"
detected_disease = detect_disease(symptom=symptom_query)

if detected_disease["info"]:
    print(f"Possible disease detected: {detected_disease['info']}")
else:
    print("No matching disease found.")

# Detection based solely on symptoms
from disease_info import DISEASE_DATABASE  # ✅ Corrige l'importation

def detect_disease_by_symptom(symptom):
    """🔎 Search for a disease by symptom."""
    return next((disease for disease in DISEASE_DATABASE.values() if symptom.lower() in disease.symptoms.lower()), None)

# Example usage:
symptom_query = "Young seedlings develop rot at the crown"
detected_disease = detect_disease_by_symptom(symptom_query)

if detected_disease:
    print(f"Possible disease detected: {detected_disease.name}")  # ✅ Affichage du nom correct
else:
    print("No matching disease found.")


# Detection using the DISEASE_DATABASE
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

# ✅ Example usage:
symptom_query = "Soft, water-soaked lesions develop without discoloration"
detected_disease = detect_disease_from_database(symptom_query)

if detected_disease:
    print(f"Possible disease detected:\n{detected_disease}")
else:
    print("⚠️ No matching disease found.")

# Example disease class update with vectors
class Disease:
    def __init__(self, name, causal_agents, distribution, symptoms, conditions, control, vectors=None):
        self.name = name
        self.causal_agents = causal_agents
        self.distribution = distribution
        self.symptoms = symptoms
        self.conditions = conditions
        self.control = control
        self.vectors = vectors or []

    def to_dict(self):
        """Returns the disease info as a dictionary."""
        return vars(self)

    def __str__(self):
        """Formats disease information for display."""
        vector_info = f"Vectors: {', '.join(self.vectors)}\n" if self.vectors else ""
        return (
            f"{self.name}\n"
            f"Causal Agents: {', '.join(self.causal_agents)}\n"
            f"Distribution: {self.distribution}\n"
            f"Symptoms: {self.symptoms}\n"
            f"Conditions: {self.conditions}\n"
            f"Control: {self.control}\n"
            f"{vector_info}"
        )

# Example disease database update
diseases = [
    Disease(
        name="Pepper Golden Mosaic Virus",
        causal_agents=["Geminivirus spp."],
        distribution="Worldwide",
        symptoms="Causes yellow mosaic patterns on leaves, reduced fruit size, and weakened plants.",
        conditions="Favored by whitefly infestations in warm, dry conditions.",
        control="Control whiteflies using insecticides, biological control, and resistant varieties.",
        vectors=["Bemisia tabaci", "Bemisia argentifolii"]
    ),
    Disease(
        name="Sinaloa Tomato Leaf Curl Virus",
        causal_agents=["Geminivirus spp."],
        distribution="Worldwide",
        symptoms="Leaf curling, stunted plants, and reduced tomato yield.",
        conditions="Spread by whiteflies; warm weather enhances transmission.",
        control="Remove infected plants and control whitefly populations effectively.",
        vectors=["Bemisia tabaci", "Bemisia argentifolii"]
    )
]
class Disease:
    def __init__(self, name, causal_agents, distribution, symptoms, conditions, control, vectors=None):
        self.name = name
        self.causal_agents = causal_agents
        self.distribution = distribution
        self.symptoms = symptoms
        self.conditions = conditions
        self.control = control
        self.vectors = vectors if vectors else []

    def __str__(self):
        """Formats disease information for display."""
        vector_info = f"Vectors: {', '.join(self.vectors)}\n" if self.vectors else ""
        return (
            f"{self.name}\n"
            f"Causal Agents: {', '.join(self.causal_agents)}\n"
            f"Distribution: {self.distribution}\n"
            f"Symptoms: {self.symptoms}\n"
            f"Conditions for Disease Development: {self.conditions}\n"
            f"Control: {self.control}\n"
            f"{vector_info}"
        )

# 📌 Example update with vectors
diseases = [
    Disease(
        name="Tomato Spotted Wilt Virus",
        causal_agents=["Tospovirus spp."],
        distribution="Worldwide",
        symptoms="Causes ringspots on leaves, stunted plants, and reduced fruit size.",
        conditions="Favored by thrips infestations and warm climates.",
        control="Use resistant crop varieties, remove infected plants, and control thrips populations.",
        vectors=["Western Flower Thrips", "Onion Thrips"]
    ),
    Disease(
        name="Peanut Bud Necrosis Virus",
        causal_agents=["Tospovirus spp."],
        distribution="Worldwide",
        symptoms="Leaf necrosis, stunted growth, and plant death.",
        conditions="Spread by thrips populations under warm conditions.",
        control="Monitor thrips activity, use insecticides, and implement field sanitation.",
        vectors=["Western Flower Thrips"]
    )
]