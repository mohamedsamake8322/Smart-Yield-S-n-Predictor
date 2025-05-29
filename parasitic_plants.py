import random

class ParasiticPlant:
    def __init__(self, name, scientific_name, affected_crops, distribution, symptoms, conditions, control):
        """Initializes a parasitic plant with all its relevant attributes."""
        self.name = name
        self.scientific_name = scientific_name
        self.affected_crops = affected_crops
        self.distribution = distribution
        self.symptoms = symptoms
        self.conditions = conditions
        self.control = control

    def __str__(self):
        """Formats plant information for display."""
        attributes = vars(self)
        return "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in attributes.items())

    def to_dict(self):
        """Returns plant details as a dictionary."""
        return vars(self)

# 📌 List of parasitic plants
parasitic_plants = [
    ParasiticPlant(
        "Dodder",
        "Cuscuta spp.",
        ["Tomatoes", "Peppers", "Eggplant", "Potatoes"],
        "Worldwide",
        "Twining leafless strands steal nutrients, causing weakness and discoloration.",
        "Thrives in well-irrigated areas and mild climates.",
        "Manual removal, pre-emergence herbicides, and crop rotation."
    ),
    ParasiticPlant(
        "Striga (Witchweed)",
        "Striga spp.",
        ["Maize", "Sorghum", "Millet", "Rice"],
        "Africa, Asia, USA",
        "Severe stunting, yellowing, and wilting due to root attachment.",
        "Prefers sandy soils with low nutrient levels.",
        "Use resistant crops, organic soil amendments, and deep plowing."
    ),
    ParasiticPlant(
        "Orobanche (Broomrape)",
        "Orobanche spp.",
        ["Tomatoes", "Legumes", "Sunflowers"],
        "Europe, Middle East, North Africa",
        "Underground root attachment causing chlorosis and poor flowering.",
        "Dry climates favor infestation; seeds remain viable for decades.",
        "Use trap crops, soil solarization, resistant varieties."
    ),
    ParasiticPlant(
        "Rafflesia",
        "Rafflesia spp.",
        ["Tetrastigma vines (wild host)"],
        "Southeast Asia",
        "Produces the largest parasitic flower; lacks leaves and stems.",
        "Strictly dependent on vine hosts for survival.",
        "No known agricultural control measures."
    ),
    ParasiticPlant(
        "Hydnora",
        "Hydnora spp.",
        ["Various desert shrubs (wild host)"],
        "Africa, Arabia",
        "Fully subterranean parasite with fleshy growths feeding on host roots.",
        "Prefers arid environments.",
        "Not managed in agricultural settings."
    )
]

# 🔎 Search function for parasitic plants
def get_parasitic_plant_by_name(name):
    """Retrieve a parasitic plant by name."""
    plant = next((p for p in parasitic_plants if p.name.lower() == name.lower()), None)
    return plant.to_dict() if plant else {"error": f"❌ '{name}' not found in database."}

# 🌱 Infestation risk prediction
def predict_infestation(crop, temperature, humidity, soil_type):
    """Predicts infestation risk based on crop and environmental conditions."""
    risk_level = random.uniform(0, 0.7)  

    for plant in parasitic_plants:
        if crop in plant.affected_crops:
            favorable_conditions = (
                ("warm" in plant.conditions and 20 <= temperature <= 35)
                or ("humid" in plant.conditions and humidity > 50)
                or ("sandy" in plant.conditions and soil_type == "sandy")
            )
            if favorable_conditions:
                risk_level += 0.3  

    return f"Predicted infestation risk for {crop}: {min(risk_level, 1):.2f} (0 = low, 1 = high)"

# ✅ Message de chargement des données
print(f"🚀 Parasitic plant database loaded successfully! ({len(parasitic_plants)} plants available)")

# 📌 Example usage of prediction system
crop = "Tomatoes"
temperature = 30  
humidity = 60  
soil_type = "sandy"
print(predict_infestation(crop, temperature, humidity, soil_type))
print("Exécution terminée avec succès !")
