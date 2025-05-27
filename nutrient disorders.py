#nutrient discoders
class Disease:
    def __init__(self, name, causal_agent, distribution, symptoms, conditions, control):
        self.name = name
        self.causal_agent = causal_agent
        self.distribution = distribution
        self.symptoms = symptoms
        self.conditions = conditions
        self.control = control

    def __str__(self):
        return "\n".join(f"{key.capitalize().replace('_', ' ')}: {value}" for key, value in vars(self).items())

# Disease Library
diseases = [
    Disease(
        "Nutrient Disorders",
        "Insufficient or excessive nutrients",
        "Worldwide",
        "\n".join([
            "Nitrogen (N): Light-green leaves, small fruits.",
            "Phosphorus (P): Dark-green small leaves.",
            "Potassium (K): Bronzing, burning leaf margins.",
            "Calcium (Ca): Chlorosis, necrosis, blossom-end rot.",
            "Magnesium (Mg): Interveinal chlorosis.",
            "Sulfur (S): Light-green spindly leaves."
        ]),
        "Most common in acidic or alkaline soils; aggravated by compaction, excess moisture, or fertilizer imbalance.",
        "Conduct soil and foliar analyses, adjust soil pH, and maintain balanced fertilization."
    ),
    Disease(
        "Salt Toxicity",
        "Excessive salts",
        "Worldwide",
        "Stunted growth, leaf tip burn, marginal chlorosis, scorched leaves, necrotic roots.",
        "Occurs with high soil or irrigation water salinity, accumulation due to cycles of wetting and drying.",
        "Test salt levels, use amendments like gypsum, manage fertilizer use, and optimize irrigation techniques."
    ),
    Disease(
        "Stip",
        "Physiological",
        "Worldwide",
        "Gray to black spots on fruits, worsens as fruit matures.",
        "Likely caused by calcium imbalance, common in older open-pollinated varieties during short, cool days.",
        "Use resistant hybrids, apply calcium, avoid susceptible varieties in fall."
    ),
    Disease(
        "Sunscald",
        "Environmental",
        "Worldwide",
        "Wrinkled, lighter-colored fruit tissue, turns paper-like and black from fungal colonization.",
        "Occurs when fruit is exposed to direct sunlight due to defoliation or stem breakage.",
        "Encourage healthy foliage, shade plants in high temperatures, use disease-resistant varieties."
    )
]

# Display Disease Information
for disease in diseases:
    print(disease)
