class InsectPest:
    def __init__(self, name, scientific_name, affected_crops, distribution, damage, transmission, control_methods):
        """Initializes an insect pest with all its relevant attributes."""
        self.name = name
        self.scientific_name = scientific_name
        self.affected_crops = affected_crops
        self.distribution = distribution
        self.damage = damage
        self.transmission = transmission
        self.control_methods = control_methods

    def __str__(self):
        """Formats pest information for display."""
        attributes = vars(self)
        return "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in attributes.items())

    def to_dict(self):
        """Returns pest details as a dictionary."""
        return vars(self)

# 📌 Unified List of Insect Pests
insect_pests = [
    InsectPest(
        "Aphids", ["Myzus persicae", "Macrosiphum euphorbiae"],
        ["Peppers", "Eggplants", "Tomatoes", "Potatoes"], "Worldwide",
        "Sucks plant sap, leading to stunting and deformation.",
        "Vector of Cucumber Mosaic Virus, Alfalfa Mosaic Virus, Pepper Mottle Virus, and Potato Virus Y.",
        "Use reflective mulches, biological controls, insecticide sprays."
    ),
    InsectPest(
        "Beet Leafhopper", "Circulifer tenellus",
        ["Tomatoes", "Peppers", "Beets"], "Worldwide, arid regions",
        "Causes leaf curling, yellowing, fruit distortion.",
        "Vector of Beet Curly Top Virus (BCTV).",
        "Manage weeds, remove infected plants, use resistant varieties."
    ),
    InsectPest(
        "Eggplant Fruit and Shoot Borer", "Leucinodes orbonalis",
        ["Eggplant"], "South and Southeast Asia",
        "Larvae cause wilting of shoots and make fruits unmarketable. Severe infestations can result in 100% yield loss.",
        "Not a known vector for viruses.",
        "Practice good crop sanitation, use pheromone traps, apply insecticides judiciously, and remove infested shoots."
    )
]

# 🔎 Pest Search Function
def get_insect_by_name(name):
    """Search for an insect pest by name."""
    pest = next((p for p in insect_pests if p.name.lower() == name.lower()), None)
    return pest.to_dict() if pest else {"error": f"❌ Pest '{name}' not found in database."}

# ✅ Indicateur de chargement des données
print(f"🚀 Insect pest database loaded successfully! ({len(insect_pests)} pests available)")

# 📌 Definition of NematodeDisease Class
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

# 📌 Unified List of Nematode Diseases
nematode_diseases = [
    NematodeDisease(
        "Root-Knot Nematodes", ["Meloidogyne incognita", "Meloidogyne javanica", "Meloidogyne arenaria"],
        "Worldwide", "Causes bead-like galls on roots, stunted growth, nutrient deficiency, wilting.",
        "Thrives in warm climates with sandy soil.",
        "Fumigation, crop rotation with non-hosts, resistant plant varieties."
    ),
    NematodeDisease(
        "Lesion Nematodes", ["Pratylenchus spp."],
        "Worldwide", "Causes root decay, blackened lesions, reduced water uptake, stunted growth.",
        "Favored by moist soil and high temperatures.",
        "Crop rotation, organic soil amendments, nematicides."
    )
]

# 🔎 Nematode Disease Search Function
def get_nematode_disease_by_name(name):
    """Search for a nematode disease by name."""
    disease = next((d for d in nematode_diseases if d.name.lower() == name.lower()), None)
    return disease.to_dict() if disease else {"error": f"❌ Disease '{name}' not found in database."}

# 📌 Interactive system for adding new nematode diseases
def add_nematode_disease():
    """Dynamically adds a new nematode disease."""
    name = input("Disease name: ")
    causal_agents = input("Causal agents (separated by commas): ").split(", ")
    distribution = input("Geographical distribution: ")
    symptoms = input("Symptoms: ")
    conditions = input("Favorable conditions: ")
    control = input("Control methods: ")

    new_disease = NematodeDisease(name, causal_agents, distribution, symptoms, conditions, control)
    nematode_diseases.append(new_disease)
    print(f"Disease '{name}' successfully added!")

# ✅ Indicateur de chargement des données
print(f"🚀 Nematode disease database loaded successfully! ({len(nematode_diseases)} diseases available)")

print("Exécution terminée avec succès !")
