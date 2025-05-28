class Disease:
    def __init__(self, name, causal_agents, distribution, symptoms, conditions, control, vectors=None):
        """Initialize a disease object with relevant details."""
        self.name = name
        self.causal_agents = causal_agents
        self.distribution = distribution
        self.symptoms = symptoms
        self.conditions = conditions
        self.control = control
        self.vectors = vectors if vectors else []

    def __str__(self):
        """Formats disease information for display."""
        attributes = vars(self)
        return "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in attributes.items())

# 📌 Centralized disease database
DISEASE_DATABASE = {
    "Anthracnose": Disease(
        "Anthracnose",
        ["Colletotrichum capsici", "C. gloeosporioides"],
        "Worldwide",
        "Affects all above-ground parts of peppers, forming pink or orange spore masses.",
        "Warm, wet weather favors infection (20°-27°C).",
        "Use high-quality seeds, crop rotation, remove infected debris."
    ),
    "Fusarium Wilt": Disease(
        "Fusarium Wilt",
        ["Fusarium oxysporum f. sp. capsici"],
        "Worldwide",
        "Early symptoms include yellowing leaves and reddish-brown streaks in stems.",
        "Fungus survives in soil for years, spreading via irrigation and farm tools.",
        "Use raised beds, disinfect equipment, and rotate crops."
    ),
    "Cucumber Mosaic Virus": Disease(
        "Cucumber Mosaic Virus",
        ["Cucumber mosaic virus"],
        "Worldwide",
        "Causes leaf mottling, distorted fruit, and stunted growth.",
        "Spreads rapidly under warm conditions.",
        "Remove infected plants, control aphids, and use virus-free seeds."
    )
}

# 📌 List of disease names for external reference
diseases = list(DISEASE_DATABASE.keys())  # ✅ Ajout explicite de `diseases` pour éviter l’erreur d'importation

# 🔎 Efficient disease retrieval
def get_disease_info(name):
    """Search for a disease by its name."""
    return DISEASE_DATABASE.get(name, "⚠️ Disease not found.")

print("🚀 Disease database loaded successfully!")
