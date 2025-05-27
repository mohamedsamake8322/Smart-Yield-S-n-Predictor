class NematodeDisease:
    def __init__(self, name, causal_agents, distribution, symptoms, conditions, control):
        self.name = name
        self.causal_agents = causal_agents
        self.distribution = distribution
        self.symptoms = symptoms
        self.conditions = conditions
        self.control = control

    def __str__(self):
        return "\n".join(f"{key.capitalize().replace('_', ' ')}: {value}" for key, value in vars(self).items())

# Extended list of nematode-caused diseases
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
    ),
    NematodeDisease(
        "Reniform Nematodes",
        ["Rotylenchulus reniformis"],
        "Tropics and subtropics",
        "Reduces root elongation, causing poor nutrient uptake and yield reduction.",
        "Favored by wet soils with organic matter.",
        "Crop rotation, resistant varieties, nematicidal treatments."
    ),
    NematodeDisease(
        "Pin Nematodes",
        ["Paratylenchus spp."],
        "Worldwide",
        "Weak root growth and reduced plant vigor.",
        "Thrives in sandy or loamy soil.",
        "Use cover crops, organic soil amendments, reduce excessive tillage."
    ),
    NematodeDisease(
        "Spiral Nematodes",
        ["Helicotylenchus spp."],
        "Worldwide",
        "Root curling, nutrient deficiency, stunting.",
        "Prefers well-drained soil with high organic matter.",
        "Apply organic matter, improve drainage, use resistant varieties."
    ),
    NematodeDisease(
        "Stubby Root Nematodes",
        ["Trichodorus spp.", "Paratrichodorus spp."],
        "Worldwide",
        "Shortened, swollen roots, reduced water and nutrient absorption.",
        "Thrives in sandy soil with adequate moisture.",
        "Apply nematicides, improve soil health, use resistant crops."
    ),
    NematodeDisease(
        "Dagger Nematodes",
        ["Xiphinema spp."],
        "Worldwide",
        "Transmits plant viruses, causes root deformation and reduced growth.",
        "Favored by temperate climates.",
        "Use nematicides, maintain soil biodiversity, implement crop rotation."
    ),
    NematodeDisease(
        "Ring Nematodes",
        ["Criconemoides spp."],
        "Worldwide",
        "Necrotic patches on roots, affects overall plant development.",
        "Prefers sandy soils with high aeration.",
        "Organic amendments, proper soil management, deep plowing."
    ),
    NematodeDisease(
        "Burrowing Nematodes",
        ["Radopholus spp."],
        "Tropics",
        "Penetrates roots deeply, causing rotting, stunting, yield loss.",
        "Favored by warm, wet conditions.",
        "Soil fumigation, resistant cultivars, monitoring soil health."
    )
]

# Efficient search for nematode diseases
def get_nematode_disease_by_name(name):
    """Quickly retrieve a nematode disease by name."""
    return next((disease for disease in nematode_diseases if disease.name.lower() == name.lower()), None)

# Interactive system to allow users to add new diseases
def add_nematode_disease():
    """Dynamically add a new nematode-related disease."""
    name = input("Disease name: ")
    causal_agents = input("Causal agents (comma-separated): ").split(", ")
    distribution = input("Geographic distribution: ")
    symptoms = input("Symptoms: ")
    conditions = input("Favorable conditions: ")
    control = input("Control methods: ")

    new_disease = NematodeDisease(name, causal_agents, distribution, symptoms, conditions, control)
    nematode_diseases.append(new_disease)
    print(f"Disease '{name}' successfully added!")
print("Exécution terminée avec succès !")

