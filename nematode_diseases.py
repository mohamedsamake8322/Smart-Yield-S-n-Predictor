class NematodeDisease:
    def __init__(self, name, causal_agents, distribution, symptoms, conditions, control):
        """Initializes a nematode disease."""
        self.name = name
        self.causal_agents = causal_agents
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

# 📌 Extended list of nematode-caused diseases
nematode_diseases = [
    NematodeDisease(
        "Root-Knot Nematodes",
        ["Meloidogyne incognita", "Meloidogyne javanica", "Meloidogyne arenaria"],
        "Worldwide",
        "Bead-like galls on roots, stunted growth, nutrient deficiency, wilting.",
        "Thrives in warm climates with sandy soil.",
        "Fumigation, crop rotation with non-hosts, resistant plant varieties."
    ),
    NematodeDisease(
        "Lesion Nematodes",
        ["Pratylenchus spp."],
        "Worldwide",
        "Root decay, blackened lesions, reduced water uptake, stunted growth.",
        "Favored by moist soil and high temperatures.",
        "Crop rotation, organic soil amendments, nematicides."
    ),
    NematodeDisease(
        "Cyst Nematodes",
        ["Heterodera spp.", "Globodera spp."],
        "Worldwide",
        "Yellowing of leaves, stunted growth, reduced yields, cysts on roots.",
        "Thrives in clay-rich soils.",
        "Resistant crop varieties, soil solarization, avoiding monoculture."
    )
]

# 🔎 Search function for nematode diseases
def get_nematode_disease_by_name(name):
    """Retrieve a nematode disease by name."""
    disease = next((d for d in nematode_diseases if d.name.lower() == name.lower()), None)
    return disease.to_dict() if disease else {"error": f"❌ '{name}' not found in database."}

# 📌 Interactive system for adding new nematode diseases
def add_nematode_disease():
    """Dynamically adds a new nematode disease."""
    name = input("Disease name: ")
    causal_agents = input("Causal agents (comma-separated): ").split(", ")
    distribution = input("Geographic distribution: ")
    symptoms = input("Symptoms: ")
    conditions = input("Favorable conditions: ")
    control = input("Control methods: ")

    new_disease = NematodeDisease(name, causal_agents, distribution, symptoms, conditions, control)
    nematode_diseases.append(new_disease)
    print(f"✅ Disease '{name}' successfully added!")

# ✅ Message de chargement des données
print(f"🚀 Nematode disease database loaded successfully! ({len(nematode_diseases)} diseases available)")

print("Exécution terminée avec succès !")
