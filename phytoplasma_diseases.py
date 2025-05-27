import random

class PhytoplasmaDisease:
    def __init__(self, name, causal_agents, vectors, affected_crops, distribution, symptoms, conditions, control):
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

# 📌 Expanded list of phytoplasma diseases
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
    ),
    PhytoplasmaDisease(
        "Banana Wilt Phytoplasma",
        ["Banana Wilt Phytoplasma"],
        ["Planthoppers (Pentalonia nigronervosa)"],
        ["Banana"],
        "Africa, Latin America",
        "Early leaf yellowing, wilting, reduced banana bunches.",
        "Humid soil with high vector density.",
        "Control planthoppers, improve drainage, strict plantation monitoring."
    ),
    PhytoplasmaDisease(
        "Potato Witches’ Broom Phytoplasma",
        ["Potato Witches’ Broom Phytoplasma"],
        ["Leafhoppers (Bactericera cockerelli)"],
        ["Potatoes"],
        "North America, Asia",
        "Distorted foliage, underdeveloped tubers, abnormal bud growth.",
        "Warm temperatures favor infection.",
        "Use resistant varieties, remove diseased plants, treat vectors."
    ),
    PhytoplasmaDisease(
        "Citrus Greening (Huanglongbing - HLB)",
        ["Candidatus Liberibacter spp."],
        ["Asian citrus psyllid (Diaphorina citri)"],
        ["Citrus"],
        "Asia, Africa, South America",
        "Yellowing leaves, deformed fruits, progressive tree death.",
        "Warm climate, high vector presence.",
        "Remove infected trees, biological control of psyllids, plantation monitoring."
    )
]

# 🔎 Efficient phytoplasma disease search
def get_phytoplasma_disease_by_name(name):
    """Search for a phytoplasma-caused disease by name."""
    return next((disease for disease in phytoplasma_diseases if disease.name.lower() == name.lower()), "❌ Disease not found.")

# 📊 Advanced phytoplasma detection system
def detect_phytoplasma_disease(symptom, climate, soil_type):
    """Predicts which phytoplasma disease may be present based on symptoms and climatic conditions."""
    possible_diseases = []

    for disease in phytoplasma_diseases:
        if symptom.lower() in disease.symptoms.lower():
            favorable_conditions = (
                ("hot" in climate.lower() and "warm temperatures" in disease.conditions) or
                ("humid" in climate.lower() and "humidity" in disease.conditions) or
                ("dry" in climate.lower() and "dry soil" in disease.conditions)
            )
            if favorable_conditions:
                possible_diseases.append(disease.name)

    if possible_diseases:
        return f"💡 Suspected diseases: {', '.join(possible_diseases)}"
    else:
        return "❌ No matching phytoplasma disease detected based on symptoms and conditions."

# Example usage of the detection system
symptom = "Yellowing leaves and deformed fruits"
climate = "hot and humid"
soil_type = "clay"

print(detect_phytoplasma_disease(symptom, climate, soil_type))
print("Exécution terminée avec succès !")
