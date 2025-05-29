class Disease:
    def __init__(self, name, causal_agent, distribution, symptoms, conditions, control):
        """Initializes a disease."""
        self.name = name
        self.causal_agent = causal_agent
        self.distribution = distribution
        self.symptoms = symptoms if isinstance(symptoms, list) else [symptoms]
        self.conditions = conditions
        self.control = control

    def __str__(self):
        """Formats disease information for display."""
        attributes = vars(self)
        attributes["symptoms"] = "\n".join(self.symptoms)  # Convert symptom list into text
        return "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in attributes.items())

    def to_dict(self):
        """Returns disease details as a dictionary."""
        return vars(self)

# 📌 Disease Library
diseases = [
    Disease(
        "Nutrient Disorders",
        "Insufficient or excessive nutrients",
        "Worldwide",
        [
            "Nitrogen (N): Light-green leaves, small fruits.",
            "Phosphorus (P): Dark-green small leaves.",
            "Potassium (K): Bronzing, burning leaf margins.",
            "Calcium (Ca): Chlorosis, necrosis, blossom-end rot.",
            "Magnesium (Mg): Interveinal chlorosis.",
            "Sulfur (S): Light-green spindly leaves."
        ],
        "Most common in acidic or alkaline soils; aggravated by compaction, excess moisture, or fertilizer imbalance.",
        "Conduct soil and foliar analyses, adjust soil pH, and maintain balanced fertilization."
    ),
    Disease(
        "Salt Toxicity",
        "Excessive salts",
        "Worldwide",
        ["Stunted growth, leaf tip burn, marginal chlorosis, scorched leaves, necrotic roots."],
        "Occurs with high soil or irrigation water salinity, accumulation due to cycles of wetting and drying.",
        "Test salt levels, use amendments like gypsum, manage fertilizer use, and optimize irrigation techniques."
    ),
    Disease(
        "Stip",
        "Physiological",
        "Worldwide",
        ["Gray to black spots on fruits, worsens as fruit matures."],
        "Likely caused by calcium imbalance, common in older open-pollinated varieties during short, cool days.",
        "Use resistant hybrids, apply calcium, avoid susceptible varieties in fall."
    ),
    Disease(
        "Sunscald",
        "Environmental",
        "Worldwide",
        ["Wrinkled, lighter-colored fruit tissue, turns paper-like and black from fungal colonization."],
        "Occurs when fruit is exposed to direct sunlight due to defoliation or stem breakage.",
        "Encourage healthy foliage, shade plants in high temperatures, use disease-resistant varieties."
    )
]

# 🔎 Search function for nutrient disorders
def get_disease_by_name(name):
    """Retrieve a nutrient disorder by name."""
    disease = next((d for d in diseases if d.name.lower() == name.lower()), None)
    return disease.to_dict() if disease else {"error": f"❌ '{name}' not found in database."}

# ✅ Message de chargement des données
print(f"🚀 Nutrient disorder database loaded successfully! ({len(diseases)} diseases available)")

# 📌 Example usage
for disease in diseases:
    print(disease)
print("Exécution terminée avec succès !")
