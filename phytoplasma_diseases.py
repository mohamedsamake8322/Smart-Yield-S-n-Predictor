import random

class PhytoplasmaDisease:
    def __init__(self, name, causal_agents, vectors, affected_crops, distribution, symptoms, conditions, control):
        """Initializes a phytoplasma disease."""
        self.name = name
        self.causal_agents = causal_agents
        self.vectors = vectors
        self.affected_crops = affected_crops
        self.distribution = distribution
        self.symptoms = symptoms
        self.conditions = conditions
        self.control = control

    def __str__(self):
        """Formats disease information for display."""
        attributes = vars(self)
        return "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in attributes.items())

    def to_dict(self):
        """Returns disease details as a dictionary."""
        return vars(self)

# 📌 List of phytoplasma diseases
phytoplasma_diseases = [
    PhytoplasmaDisease(
        "Little Leaf Phytoplasma",
        ["Little Leaf Phytoplasma"],
        ["Cotton Leafhopper (Hishimonus phycitis)"],
        ["Eggplant"],
        "India, Bangladesh",
        "Small pale-green leaves, shortened stems, sterile flowering, significantly reduced yield.",
        "High leafhopper presence and warm/humid conditions.",
        "Remove infected plants, control leafhoppers using biological insecticides."
    ),
    PhytoplasmaDisease(
        "Stolbur Phytoplasma",
        ["Stolbur Phytoplasma"],
        ["Hyalesthes obsoletus (Leafhoppers)"],
        ["Tomatoes", "Grapevines", "Peppers"],
        "Europe, North Africa",
        "Stunting, chlorosis, abnormal flowering, underdeveloped roots.",
        "Warm and dry soil, high vector presence.",
        "Crop rotation, leafhopper control, plantation monitoring."
    ),
    PhytoplasmaDisease(
        "Aster Yellows Phytoplasma",
        ["Aster Yellows Phytoplasma"],
        ["Macrosteles quadrilineatus (Leafhoppers)"],
        ["Vegetables", "Cereals"],
        "North America, Europe",
        "Leaf deformation, sterile flowering, widespread chlorosis.",
        "High leafhopper presence, warm/humid conditions.",
        "Field monitoring, biological treatments, removal of infected plants."
    ),
    PhytoplasmaDisease(
        "Papaya Bunchy Top Phytoplasma",
        ["Papaya Bunchy Top Phytoplasma"],
        ["Myndus spp. (Leafhoppers)"],
        ["Papaya"],
        "Africa, Southeast Asia",
        "Stunted growth, deformed leaves, severe yield loss.",
        "Heat and humidity favor spread.",
        "Remove infected plants, biological control of leafhoppers."
    )
]

# 🔎 Search function for phytoplasma diseases
def get_phytoplasma_disease_by_name(name):
    """Retrieve phytoplasma disease details by name."""
    disease = next((d for d in phytoplasma_diseases if d.name.lower() == name.lower()), None)
    return disease.to_dict() if disease else {"error": f"❌ '{name}' not found in database."}

# 🌱 Prediction system for phytoplasma diseases
def detect_phytoplasma_disease(symptom, climate, soil_type):
    """Predicts which phytoplasma disease may be present based on symptoms and climatic conditions."""
    possible_diseases = []

    for disease in phytoplasma_diseases:
        if symptom.lower() in disease.symptoms.lower():
            favorable_conditions = (
                ("warm" in disease.conditions and "hot" in climate.lower())
                or ("humid" in disease.conditions and "humid" in climate.lower())
                or ("dry" in disease.conditions and "dry" in climate.lower())
            )
            if favorable_conditions:
                possible_diseases.append(disease.name)

    return {"suspected_diseases": possible_diseases} if possible_diseases else {"error": "❌ No matching disease detected."}

# ✅ Message de chargement des données
print(f"🚀 Phytoplasma disease database loaded successfully! ({len(phytoplasma_diseases)} diseases available)")

# 📌 Example usage of prediction system
symptom = "Yellowing leaves and deformed fruits"
climate = "hot and humid"
soil_type = "clay"

print(detect_phytoplasma_disease(symptom, climate, soil_type))
print("Exécution terminée avec succès !")
