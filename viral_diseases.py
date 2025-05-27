class ViralDisease:
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

# 📌 Liste optimisée des maladies virales
viral_diseases = [
    ViralDisease(
        "Alfalfa Mosaic Virus (AMV)",
        ["Alfalfa Mosaic Virus"],
        ["Aphids"],
        ["Peppers", "Tomatoes"],
        "Worldwide",
        "Bright-yellow mosaic on leaves, mottled and distorted fruits.",
        "Peppers grown near alfalfa fields where aphids are abundant.",
        "Avoid planting near alfalfa fields, rogue infected plants, and control aphid populations."
    ),
    ViralDisease(
        "Beet Curly Top Virus (BCTV)",
        ["Beet Curly Top Virus"],
        ["Beet Leafhopper (Circulifer tenellus)"],
        ["Peppers", "Tomatoes", "Beets", "Squash"],
        "Worldwide, especially in arid/semi-arid regions.",
        "Yellowing, twisting of leaves, stiff petioles, reduced fruit set.",
        "High leafhopper populations, warm temperatures favor spread.",
        "Use virus-free transplants, remove infected plants, control weeds near fields."
    ),
    ViralDisease(
        "Cucumber Mosaic Virus (CMV)",
        ["Cucumber Mosaic Virus"],
        ["Aphids"],
        ["Peppers", "Eggplant"],
        "Worldwide",
        "Narrow, distorted leaves, tip dieback, oak-leaf discoloration, defoliation.",
        "Warm conditions with aphids present; often persists in alternate hosts.",
        "Eliminate nearby weeds, use reflective mulches, apply stylet oil and insecticide sprays."
    ),
    ViralDisease(
        "Geminiviruses",
        ["Pepper Huasteco Yellow Vein Virus", "Sinaloa Tomato Leaf Curl Virus"],
        ["Whiteflies (Bemisia tabaci, B. argentifolii)"],
        ["Peppers", "Eggplant"],
        "Worldwide",
        "Yellow vein-etching, distorted leaves, stunted plants, deformed fruits.",
        "Hot climates with overlapping cropping systems; whiteflies are main vectors.",
        "Exclude whiteflies using netting, apply systemic insecticides early, destroy infected crops."
    ),
    ViralDisease(
        "Pepper Mottle Virus (PepMoV)",
        ["Pepper Mottle Virus"],
        ["Aphids"],
        ["Peppers"],
        "Southern United States, California, Mexico, Central America",
        "Systemic mottle, distortion, stunted plants. Greenhouse plants show vein-clearing followed by chlorotic mottle. "
        "Fruit may be distorted, mottled, and small.",
        "Transmitted by aphids and mechanically through handling, pruning, and staking.",
        "Remove infected crop residues and weeds, use reflective mulches, apply stylet oils and insecticide sprays."
    ),
    ViralDisease(
        "Potato Virus X (PVX)",
        ["Potato Virus X"],
        ["Mechanical transmission (no insect vectors)"],
        ["Potatoes", "Tomatoes", "Eggplants"],
        "Worldwide",
        "Necrotic spots, distortion, ringspots, small leaves, stunted plants, bushy growth, reduced yield.",
        "Spread mainly through transplanting, pruning, grafting, and equipment contamination.",
        "Sanitize tools, rogue infected plants, avoid planting peppers or eggplants after potatoes."
    ),
    ViralDisease(
        "Potato Virus Y (PVY)",
        ["Potato Virus Y"],
        ["Aphids"],
        ["Potatoes", "Peppers", "Tomatoes"],
        "Worldwide",
        "Mottling, crinkling, yellowing, leaf necrosis, reduced yield, distorted fruit.",
        "Spread rapidly by aphids; survives in alternate hosts.",
        "Control aphid populations, plant resistant varieties, eliminate infected plants."
    )
]

# 🔎 Recherche efficace des maladies virales
def get_viral_disease_by_name(name):
    """Recherche une maladie virale par son nom."""
    return next((disease for disease in viral_diseases if disease.name.lower() == name.lower()), "⚠️ Disease not found.")

class ViralDisease:
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

# 📌 Optimized viral disease list
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
        "Tobamoviruses",
        ["Tobacco Mosaic Virus (TMV)", "Tomato Mosaic Virus (ToMV)", "Pepper Mild Mottle Virus (PMMV)"],
        ["Mechanical transmission (no insect vectors)"],
        ["Peppers", "Tomatoes", "Eggplants", "Tobacco"],
        "Worldwide",
        "Chlorotic mosaic, distortion, necrosis, defoliation, small disfigured fruit.",
        "Extensive spread through handling, tools, pollination, cultural practices.",
        "Strict sanitation measures, seed treatment, resistant varieties."
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
    ),
    ViralDisease(
        "Maize Dwarf Mosaic Virus (MDMV)",
        ["Maize Dwarf Mosaic Virus"],
        ["Aphids"],
        ["Corn", "Sorghum"],
        "Worldwide",
        "Chlorotic mosaic, red streaks on leaves, slow growth.",
        "Presence of aphids, warm and humid conditions favor spread.",
        "Use resistant varieties, apply insecticides to young plants, eliminate alternate hosts."
    ),
    ViralDisease(
        "Barley Yellow Dwarf Virus (BYDV)",
        ["Barley Yellow Dwarf Virus"],
        ["Aphids"],
        ["Wheat", "Barley", "Oats"],
        "Worldwide",
        "Leaf yellowing, stunted growth, young plant mortality.",
        "Cool to moderate climates with high aphid density.",
        "Plant resistant varieties, apply preventive insecticides, avoid early planting."
    ),
    ViralDisease(
        "Soybean Mosaic Virus (SMV)",
        ["Soybean Mosaic Virus"],
        ["Aphids"],
        ["Soybeans"],
        "Worldwide",
        "Mosaic on leaves, reduced pod development, lower yield.",
        "Presence of aphids and warm, humid conditions favor infection.",
        "Use tested seeds, manage aphid populations, remove infected plants."
    )
]

# 🔎 Efficient viral disease search
def get_viral_disease_by_name(name):
    """Search for a viral disease by name."""
    return next((disease for disease in viral_diseases if disease.name.lower() == name.lower()), "⚠️ Disease not found.")
import random

# 📌 Advanced viral disease detection system
def detect_viral_disease(symptoms, climate, soil_type):
    """Predicts which viral disease may be present based on symptoms and climatic conditions."""
    possible_diseases = []

    for disease in viral_diseases:
        if any(symptom.lower() in disease.symptoms.lower() for symptom in symptoms):
            favorable_conditions = (
                ("hot" in climate.lower() and "hot" in disease.conditions) or
                ("humid" in climate.lower() and "humid" in disease.conditions) or
                ("dry" in climate.lower() and "dry" in disease.conditions)
            )
            if favorable_conditions:
                possible_diseases.append(disease.name)

    if possible_diseases:
        return f"💡 Suspected diseases: {', '.join(possible_diseases)}"
    else:
        return "❌ No matching viral disease detected based on symptoms and conditions."

# Example usage of the detection system
symptoms = ["Leaf yellowing", "Stunted growth"]
climate = "cool and humid"
soil_type = "clay"

print(detect_viral_disease(symptoms, climate, soil_type))

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

    return "❌ Disease not found."

# Example usage of the predictive model
disease_name = "Maize Dwarf Mosaic Virus"
temperature = 28  # °C
humidity = 65  # %
aphid_population = 600  # Estimated aphid count

print(predict_viral_risk(disease_name, temperature, humidity, aphid_population))