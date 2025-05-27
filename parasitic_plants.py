import random

class ParasiticPlant:
    def __init__(self, name, scientific_name, affected_crops, distribution, symptoms, conditions, control):
        self.name = name
        self.scientific_name = scientific_name
        self.affected_crops = affected_crops
        self.distribution = distribution
        self.symptoms = symptoms
        self.conditions = conditions
        self.control = control

    def __str__(self):
        return "\n".join(f"{key.capitalize().replace('_', ' ')}: {value}" for key, value in vars(self).items())

# Extended list of parasitic plants
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

# Efficient search for parasitic plants
def get_parasitic_plant_by_name(name):
    """Quickly retrieve a parasitic plant by name."""
    return next((plant for plant in parasitic_plants if plant.name.lower() == name.lower()), None)

# Infestation prediction system
def predict_infestation(crop, temperature, humidity, soil_type):
    """Predicts the risk of infestation based on climatic and plant conditions."""
    risk_level = random.uniform(0, 1)  

    for plant in parasitic_plants:
        if crop in plant.affected_crops:
            favorable_conditions = (
                (20 <= temperature <= 35 if "warm climates" in plant.conditions else True)
                and (humidity > 50 if "humid environments" in plant.conditions else True)
                and (soil_type == "sandy" if "sandy soils" in plant.conditions else True)
            )

            if favorable_conditions:
                risk_level += 0.3  

    return f"Predicted infestation risk for {crop}: {min(risk_level, 1):.2f} (0 = low, 1 = high)"

# Example usage of prediction system
crop = "Tomatoes"
temperature = 30  
humidity = 60  
soil_type = "sandy"
print(predict_infestation(crop, temperature, humidity, soil_type))
