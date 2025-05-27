class InsectPest:
    def __init__(self, name, scientific_name, description, damage, transmission, affected_plants, control_methods):
        """Initializes an insect pest with all its relevant attributes."""
        self.name = name
        self.scientific_name = scientific_name
        self.description = description
        self.damage = damage
        self.transmission = transmission
        self.affected_plants = affected_plants
        self.control_methods = control_methods

    def __str__(self):
        """Formats pest information for display."""
        return (
            f"{self.name} ({self.scientific_name})\n"
            f"Description: {self.description}\n"
            f"Damage: {self.damage}\n"
            f"Virus Transmission: {self.transmission}\n"
            f"Affected Plants: {', '.join(self.affected_plants)}\n"
            f"Control Methods: {self.control_methods}"
        )

# 📌 List of insect pests
insect_pests = [
    InsectPest(
        "Aphid",
        "Macrosiphum euphorbiae, Myzus persicae",
        "Small, pear-shaped insects that reproduce rapidly.",
        "Causes chlorosis, leaf curling, and flower abscission.",
        "Vectors viruses such as Cucumber mosaic virus and Potato leaf roll virus.",
        ["Peppers", "Eggplants"],
        "Use insecticidal soaps, natural predators like ladybugs, and crop rotation."
    ),
    InsectPest(
        "Epilachna Beetle",
        "Epilachna spp.",
        "Eggplant pest found in Asia, skeletonizes leaf tissue.",
        "Larvae cause significant yield reduction.",
        "Not a known virus vector.",
        ["Eggplant"],
        "Use insecticides, introduce natural predators, and implement crop rotation."
    ),
    InsectPest(
        "Leafhoppers",
        "Circulifer tenellus (Beet Leafhopper), Hishimonus phycitis (Cotton Leafhopper)",
        "Small wedge-shaped insects that feed on phloem.",
        "Causes yellowing, necrotic spots, and yield reduction.",
        "Transmits viruses like Beet curly top virus and Little leaf disease.",
        ["Peppers", "Eggplant"],
        "Use row covers, maintain weed control, and apply insecticides."
    ),
    InsectPest(
        "Two-Spotted Spider Mite",
        "Tetranychus urticae",
        "Creates fine webbing on infested leaves.",
        "Causes discoloration and reduced plant vigor.",
        "Not a known virus vector.",
        ["Eggplant", "Peppers"],
        "Apply miticides, insecticidal soaps, and water sprays."
    ),
    InsectPest(
        "Sweet Potato Whitefly / Silverleaf Whitefly",
        ["Bemisia tabaci", "Bemisia argentifolii"],
        "Serious pest attacking peppers and eggplants.",
        "Leads to stunting and defoliation.",
        "Vectors geminiviruses like Tomato yellow mosaic virus.",
        ["Peppers", "Eggplant"],
        "Use resistant crop varieties, insecticidal treatments, and remove infested plants."
    )
]

# 🔎 Pest search function
def get_insect_by_name(name):
    """Search for an insect pest by name."""
    return next((pest for pest in insect_pests if pest.name.lower() == name.lower()), None)

# Example usage
if __name__ == "__main__":
    insect_name = input("🔍 Enter the name of the pest to search for: ")
    result = get_insect_by_name(insect_name)

    if result:
        print("\n🦟 Pest found:\n")
        print(result)
    else:
        print("❌ No pest found under this name.")

# Liste des insectes nuisibles et vecteurs de virus
insect_pests = [
    InsectPest(
        name="Western Flower Thrips",
        scientific_name="Frankliniella occidentalis",
        description=(
            "Native to the western USA, but now spread worldwide. Winged and mobile, reproduces rapidly without mating."
        ),
        damage=(
            "Feeds on young leaves and flowers, causing twisted and cupped leaves in peppers, browning in eggplant. "
            "Egg deposition causes scarring and discoloration in fruit."
        ),
        transmission="Vectors Tomato spotted wilt virus and Peanut bud necrosis virus.",
        affected_plants=["Peppers", "Eggplant"],
        control_methods="Use insecticidal sprays, biological control, and monitor populations with white paper shake tests."
    ),
    InsectPest(
        name="Onion Thrips",
        scientific_name="Thrips tabaci",
        description="Occurs worldwide and affects various crops including peppers and eggplant.",
        damage="Causes leaf distortion and scarring, leading to reduced plant health and yield.",
        transmission="Vectors Tomato spotted wilt virus.",
        affected_plants=["Peppers", "Eggplant"],
        control_methods="Maintain good field sanitation, use reflective mulches, and control populations early."
    ),
    InsectPest(
        name="Greenhouse Thrips",
        scientific_name="Heliothrips haemorrhoidalis",
        description="Found in greenhouses worldwide, attacking ornamental and vegetable plants.",
        damage="Leads to browning and leaf damage, affecting overall crop vitality.",
        transmission="Not a known vector for viruses.",
        affected_plants=["Peppers", "Eggplant", "Various greenhouse crops"],
        control_methods="Use natural predators, insecticidal soaps, and maintain proper ventilation."
    )
]
def get_insect_by_name(name):
    """Recherche un insecte nuisible par son nom."""
    for insect in insect_pests:
        if insect.name.lower() == name.lower():
            return insect
    return None
class InsectPest:
    def __init__(self, name, scientific_name, description, damage, transmission, affected_plants, control_methods):
        self.name = name
        self.scientific_name = scientific_name
        self.description = description
        self.damage = damage
        self.transmission = transmission
        self.affected_plants = affected_plants
        self.control_methods = control_methods

    def __str__(self):
        return (
            f"{self.name} ({self.scientific_name})\n"
            f"Description: {self.description}\n"
            f"Damage: {self.damage}\n"
            f"Virus Transmission: {self.transmission}\n"
            f"Affected Plants: {', '.join(self.affected_plants)}\n"
            f"Control Methods: {self.control_methods}"
        )
# Ajout de Eggplant Fruit and Shoot Borer
insect_pests = [
    InsectPest(
        name="Eggplant Fruit and Shoot Borer",
        scientific_name="Leucinodes orbonalis",
        description=(
            "A major pest of eggplant in South and Southeast Asia. Adult moths deposit eggs on new leaves, "
            "and larvae bore into young shoots and fruits."
        ),
        damage=(
            "Larvae cause wilting of shoots and make fruits unmarketable. Severe infestations can result in 100% yield loss."
        ),
        transmission="Not a known vector for viruses.",
        affected_plants=["Eggplant"],
        control_methods=(
            "Practice good crop sanitation, use pheromone traps, apply insecticides judiciously, "
            "and remove/destroy infested shoots and crop residues."
        )
    )
]
def get_insect_by_name(name):
    """Recherche un insecte nuisible par son nom."""
    for insect in insect_pests:
        if insect.name.lower() == name.lower():
            return insect
    return None
class NematodeDisease:
    def __init__(self, name, causal_agents, distribution, symptoms, conditions, control):
        self.name = name
        self.causal_agents = causal_agents
        self.distribution = distribution
        self.symptoms = symptoms
        self.conditions = conditions
        self.control = control
    def __str__(self):
        return (
            f"{self.name}\n"
            f"Causal Agents: {', '.join(self.causal_agents)}\n"
            f"Distribution: {self.distribution}\n"
            f"Symptoms: {self.symptoms}\n"
            f"Conditions for Disease Development: {self.conditions}\n"
            f"Control: {self.control}"
        )
# Liste étendue des maladies causées par les nématodes
nematode_diseases = [
    NematodeDisease(
        name="Root-Knot Nematodes",
        causal_agents=["Meloidogyne incognita", "Meloidogyne javanica", "Meloidogyne arenaria"],
        distribution="Worldwide",
        symptoms="Causes bead-like galls on roots, stunted growth, nutrient deficiency, wilting.",
        conditions="Thrives in warm climates with sandy soil.",
        control="Fumigation, crop rotation with non-hosts, resistant plant varieties."
    ),
    NematodeDisease(
        name="Lesion Nematodes",
        causal_agents=["Pratylenchus spp."],
        distribution="Worldwide",
        symptoms="Causes root decay, blackened lesions, reduced water uptake, stunted growth.",
        conditions="Favored by moist soil and high temperatures.",
        control="Crop rotation, organic soil amendments, nematicides."
    ),
    NematodeDisease(
        name="Cyst Nematodes",
        causal_agents=["Heterodera spp.", "Globodera spp."],
        distribution="Worldwide",
        symptoms="Causes yellowing of leaves, stunted growth, reduced yields, cysts on roots.",
        conditions="Thrives in clay-rich soils.",
        control="Resistant crop varieties, soil solarization, avoiding monoculture."
    ),
    NematodeDisease(
        name="Reniform Nematodes",
        causal_agents=["Rotylenchulus reniformis"],
        distribution="Tropics and subtropics",
        symptoms="Reduces root elongation, causing poor nutrient uptake and yield reduction.",
        conditions="Favored by wet soils with organic matter.",
        control="Crop rotation, resistant varieties, nematicidal treatments."
    ),
    NematodeDisease(
        name="Pin Nematodes",
        causal_agents=["Paratylenchus spp."],
        distribution="Worldwide",
        symptoms="Causes weak root growth and reduced plant vigor.",
        conditions="Thrives in sandy or loamy soil.",
        control="Use cover crops, organic soil amendments, reduce excessive tillage."
    ),
    NematodeDisease(
        name="Spiral Nematodes",
        causal_agents=["Helicotylenchus spp."],
        distribution="Worldwide",
        symptoms="Causes root curling, nutrient deficiency, stunting.",
        conditions="Prefers well-drained soil with high organic matter.",
        control="Apply organic matter, improve drainage, use resistant varieties."
    ),
    NematodeDisease(
        name="Stubby Root Nematodes",
        causal_agents=["Trichodorus spp.", "Paratrichodorus spp."],
        distribution="Worldwide",
        symptoms="Shortened, swollen roots, reduced water and nutrient absorption.",
        conditions="Thrives in sandy soil with adequate moisture.",
        control="Apply nematicides, improve soil health, use resistant crops."
    ),
    NematodeDisease(
        name="Dagger Nematodes",
        causal_agents=["Xiphinema spp."],
        distribution="Worldwide",
        symptoms="Transmits plant viruses, causes root deformation and reduced growth.",
        conditions="Favored by temperate climates.",
        control="Use nematicides, maintain soil biodiversity, implement crop rotation."
    ),
    NematodeDisease(
        name="Ring Nematodes",
        causal_agents=["Criconemoides spp."],
        distribution="Worldwide",
        symptoms="Causes necrotic patches on roots, affects overall plant development.",
        conditions="Prefers sandy soils with high aeration.",
        control="Organic amendments, proper soil management, deep plowing."
    ),
    NematodeDisease(
        name="Burrowing Nematodes",
        causal_agents=["Radopholus spp."],
        distribution="Tropics",
        symptoms="Penetrates roots deeply, causing rotting, stunting, yield loss.",
        conditions="Favored by warm, wet conditions.",
        control="Soil fumigation, resistant cultivars, monitoring soil health."
    )
]

def get_nematode_disease_by_name(name):
    """Recherche une maladie des nématodes par son nom."""
    for disease in nematode_diseases:
        if disease.name.lower() == name.lower():
            return disease
    return None


class InsectPest:
    def __init__(self, name, scientific_name, description, damage, transmission, affected_plants, control_methods):
        self.name = name
        self.scientific_name = scientific_name
        self.description = description
        self.damage = damage
        self.transmission = transmission
        self.affected_plants = affected_plants
        self.control_methods = control_methods

    def __str__(self):
        """Formats pest information for display."""
        attributes = vars(self)
        return "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in attributes.items())

# 📌 Optimized insect pest list
insect_pests = [
    InsectPest(
        "Western Flower Thrips",
        "Frankliniella occidentalis",
        "Native to the western USA, but now spread worldwide. Winged and mobile, reproduces rapidly without mating.",
        "Feeds on young leaves and flowers, causing twisted and cupped leaves in peppers, browning in eggplant. "
        "Egg deposition causes scarring and discoloration in fruit.",
        "Vectors Tomato spotted wilt virus and Peanut bud necrosis virus.",
        ["Peppers", "Eggplant"],
        "Use insecticidal sprays, biological control, and monitor populations with white paper shake tests."
    ),
    InsectPest(
        "Onion Thrips",
        "Thrips tabaci",
        "Occurs worldwide and affects various crops including peppers and eggplant.",
        "Causes leaf distortion and scarring, leading to reduced plant health and yield.",
        "Vectors Tomato spotted wilt virus.",
        ["Peppers", "Eggplant"],
        "Maintain good field sanitation, use reflective mulches, and control populations early."
    ),
    InsectPest(
        "Greenhouse Thrips",
        "Heliothrips haemorrhoidalis",
        "Found in greenhouses worldwide, attacking ornamental and vegetable plants.",
        "Leads to browning and leaf damage, affecting overall crop vitality.",
        "Not a known vector for viruses.",
        ["Peppers", "Eggplant", "Various greenhouse crops"],
        "Use natural predators, insecticidal soaps, and maintain proper ventilation."
    ),
    InsectPest(
        "Eggplant Fruit and Shoot Borer",
        "Leucinodes orbonalis",
        "A major pest of eggplant in South and Southeast Asia. Adult moths deposit eggs on new leaves, "
        "and larvae bore into young shoots and fruits.",
        "Larvae cause wilting of shoots and make fruits unmarketable. Severe infestations can result in 100% yield loss.",
        "Not a known vector for viruses.",
        ["Eggplant"],
        "Practice good crop sanitation, use pheromone traps, apply insecticides judiciously, "
        "and remove/destroy infested shoots and crop residues."
    )
]

# 🔎 Efficient pest search
def get_insect_by_name(name):
    """Search for an insect pest by name."""
    return next((pest for pest in insect_pests if pest.name.lower() == name.lower()), "❌ Pest not found.")

class NematodeDisease:
    def __init__(self, name, causal_agents, distribution, symptoms, conditions, control):
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

# 📌 Expanded nematode disease list
nematode_diseases = [
    NematodeDisease(
        "Root-Knot Nematodes",
        ["Meloidogyne incognita", "Meloidogyne javanica", "Meloidogyne arenaria"],
        "Worldwide",
        "Causes bead-like galls on roots, stunted growth, nutrient deficiency, wilting.",
        "Thrives in warm climates with sandy soil.",
        "Fumigation, crop rotation with non-hosts, resistant plant varieties."
    ),
    NematodeDisease(
        "Lesion Nematodes",
        ["Pratylenchus spp."],
        "Worldwide",
        "Causes root decay, blackened lesions, reduced water uptake, stunted growth.",
        "Favored by moist soil and high temperatures.",
        "Crop rotation, organic soil amendments, nematicides."
    ),
    NematodeDisease(
        "Cyst Nematodes",
        ["Heterodera spp.", "Globodera spp."],
        "Worldwide",
        "Causes yellowing of leaves, stunted growth, reduced yields, cysts on roots.",
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
    )
]
# 🔎 Efficient nematode disease search
def get_nematode_disease_by_name(name):
    """Search for a nematode disease by name."""
    return next((disease for disease in nematode_diseases if disease.name.lower() == name.lower()), "❌ Disease not found.")
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

# 📌 Updating viral disease vectors
insect_pests.append(
    InsectPest(
        name="Beet Leafhopper",
        scientific_name="Circulifer tenellus",
        affected_crops=["Tomatoes", "Peppers", "Beets"],
        distribution="Worldwide, arid regions",
        damage="Causes leaf curling, yellowing, fruit distortion.",
        transmission="Vector of Beet Curly Top Virus (BCTV).",
        control_methods="Manage weeds, remove infected plants, use resistant varieties."
    )
)

insect_pests.append(
    InsectPest(
        name="Sweet Potato Whitefly",
        scientific_name="Bemisia tabaci",
        affected_crops=["Tomatoes", "Peppers", "Eggplant"],
        distribution="Worldwide",
        damage="Reduces growth, weakens plants, transmits multiple viruses.",
        transmission="Vector of Geminiviruses (Pepper Golden Mosaic Virus).",
        control_methods="Exclude whiteflies with netting, use systemic insecticides."
    )
)
insect_pests.append(
    InsectPest(
        name="Aphids",
        scientific_name="Myzus persicae, Macrosiphum euphorbiae",
        affected_crops=["Peppers", "Eggplant", "Tomatoes"],
        distribution="Worldwide",
        damage="Transmits various viruses, causes leaf curling and deformation.",
        transmission="Vector of Cucumber Mosaic Virus and Alfalfa Mosaic Virus.",
        control_methods="Use reflective mulches, insecticidal sprays, biological controls."
    )
)
# 🔎 Efficient insect pest search
def get_insect_by_name(name):
    """Search for an insect pest by name."""
    return next((insect for insect in insect_pests if insect.name.lower() == name.lower()), "❌ Pest not found.")
# 📌 Adding new viral disease vectors
insect_pests.extend([
    InsectPest(
        name="Aphids",
        scientific_name=["Myzus persicae", "Macrosiphum euphorbiae"],
        affected_crops=["Peppers", "Eggplants", "Tomatoes", "Potatoes"],
        distribution="Worldwide",
        damage="Sucks plant sap, leading to stunting and deformation.",
        transmission="Vector of Cucumber Mosaic Virus, Alfalfa Mosaic Virus, Pepper Mottle Virus, and Potato Virus Y.",
        control_methods="Use reflective mulches, biological controls, insecticide sprays."
    )
])
# 🔎 Pest search function
def get_insect_by_name(name):
    """Search for an insect pest by name."""
    return next((insect for insect in insect_pests if insect.name.lower() == name.lower()), "❌ Pest not found.")
print("Exécution terminée avec succès !")
