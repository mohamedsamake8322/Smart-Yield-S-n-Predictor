import random

class ViralDisease:
    def __init__(self, name, causal_agents, vectors, affected_crops, distribution, symptoms, conditions, control):
        """Initializes a viral disease."""
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

# 📌 List of viral diseases
viral_diseases = [
    ViralDisease(
        "Potato Virus Y (PVY)",
        ["Potato Virus Y"],
        ["Aphids"],
        ["Potatoes", "Peppers", "Tomatoes", "Eggplants"],
        "Worldwide",
        "Mottling, vein-banding, distortion, necrotic stems, discolored small fruits.",
        "Spread rapidly by aphids; survives in alternate hosts.",
        "Remove infected plants, use reflective mulches, control aphid populations."
    ),
    ViralDisease(
        "Tobacco Etch Virus (TEV)",
        ["Tobacco Etch Virus"],
        ["Aphids"],
        ["Peppers", "Tomatoes", "Tobacco"],
        "North and South America",
        "Mottling, distortion, stunting, wilting in hot peppers, chlorotic streaks on fruit.",
        "Spread by aphids moving from alternate hosts into fields.",
        "Use resistant varieties, remove infected plants, apply stylet oils and insecticide sprays."
    ),
    ViralDisease(
        "Tospoviruses",
        ["Tomato Spotted Wilt Virus (TSWV)", "Peanut Bud Necrosis Virus (PBNV)"],
        ["Thrips (Frankliniella occidentalis, Thrips palmi)"],
        ["Peppers", "Eggplants"],
        "Worldwide",
        "Yellow/necrotic concentric rings, mosaic with chlorotic spots, fruit deformation.",
        "High thrips populations, temperatures above 22°C.",
        "Use virus-free transplants, control thrips populations, remove infected plants."
    )
]

# 🔎 Search function for viral diseases
def get_viral_disease_by_name(name):
    """Retrieve viral disease details by name."""
    disease = next((d for d in viral_diseases if d.name.lower() == name.lower()), None)
    return disease.to_dict() if disease else {"error": f"❌ '{name}' not found in database."}

# 🌱 Prediction system for viral diseases
def detect_viral_disease(symptoms, climate, soil_type):
    """Predicts which viral disease may be present based on symptoms and climatic conditions."""
    possible_diseases = []

    for disease in viral_diseases:
        if any(symptom.lower() in disease.symptoms.lower() for symptom in symptoms):
            favorable_conditions = (
                ("hot" in disease.conditions and "hot" in climate.lower())
                or ("humid" in disease.conditions and "humid" in climate.lower())
                or ("dry" in disease.conditions and "dry" in climate.lower())
            )
            if favorable_conditions:
                possible_diseases.append(disease.name)

    return {"suspected_diseases": possible_diseases} if possible_diseases else {"error": "❌ No matching disease detected."}

# 📊 Predictive model for estimating infection risk based on weather conditions
def predict_viral_risk(disease_name, temperature, humidity, aphid_population):
    """Predicts the infection risk for a given viral disease based on environmental conditions."""
    base_risk = random.uniform(0.2, 0.8)

    for disease in viral_diseases:
        if disease.name.lower() == disease_name.lower():
            risk_factor = base_risk
            if temperature > 25 and "hot" in disease.conditions:
                risk_factor += 0.2
            if humidity > 60 and "humid" in disease.conditions:
                risk_factor += 0.15
            if aphid_population > 500 and "Aphids" in disease.vectors:
                risk_factor += 0.3

            risk_factor = min(risk_factor, 1)  # Ensure risk does not exceed 1
            return f"🔍 Estimated infection risk for {disease_name}: {risk_factor:.2f} (0 = low, 1 = high)"

    return {"error": "❌ Disease not found."}

# ✅ Message de chargement des données
print(f"🚀 Viral disease database loaded successfully! ({len(viral_diseases)} diseases available)")

# 📌 Example usage of prediction system
symptoms = ["Leaf yellowing", "Stunted growth"]
climate = "cool and humid"
soil_type = "clay"

print(detect_viral_disease(symptoms, climate, soil_type))

# Example usage of the predictive model
disease_name = "Potato Virus Y"
temperature = 28  # °C
humidity = 65  # %
aphid_population = 600  # Estimated aphid count

print(predict_viral_risk(disease_name, temperature, humidity, aphid_population))
print("Exécution terminée avec succès !")
