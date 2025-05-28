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

# 📌 Unified List of Insect Pests (Avoiding Duplicate Definitions)
insect_pests = [
    InsectPest("Aphid", "Macrosiphum euphorbiae, Myzus persicae", "Small, pear-shaped insects that reproduce rapidly.",
               "Causes chlorosis, leaf curling, and flower abscission.",
               "Vectors viruses such as Cucumber mosaic virus and Potato leaf roll virus.",
               ["Peppers", "Eggplants"], "Use insecticidal soaps, natural predators like ladybugs, and crop rotation."),

    InsectPest("Epilachna Beetle", "Epilachna spp.", "Eggplant pest found in Asia, skeletonizes leaf tissue.",
               "Larvae cause significant yield reduction.", "Not a known virus vector.", ["Eggplant"],
               "Use insecticides, introduce natural predators, and implement crop rotation."),

    InsectPest("Leafhoppers", "Circulifer tenellus, Hishimonus phycitis", "Small wedge-shaped insects that feed on phloem.",
               "Causes yellowing, necrotic spots, and yield reduction.",
               "Transmits viruses like Beet curly top virus and Little leaf disease.", ["Peppers", "Eggplant"],
               "Use row covers, maintain weed control, and apply insecticides."),

    InsectPest("Two-Spotted Spider Mite", "Tetranychus urticae", "Creates fine webbing on infested leaves.",
               "Causes discoloration and reduced plant vigor.", "Not a known virus vector.", ["Eggplant", "Peppers"],
               "Apply miticides, insecticidal soaps, and water sprays."),

    InsectPest("Sweet Potato Whitefly / Silverleaf Whitefly", ["Bemisia tabaci", "Bemisia argentifolii"],
               "Serious pest attacking peppers and eggplants.", "Leads to stunting and defoliation.",
               "Vectors geminiviruses like Tomato yellow mosaic virus.", ["Peppers", "Eggplant"],
               "Use resistant crop varieties, insecticidal treatments, and remove infested plants."),

    InsectPest("Eggplant Fruit and Shoot Borer", "Leucinodes orbonalis",
               "A major pest of eggplant in South and Southeast Asia. Larvae bore into young shoots and fruits.",
               "Causes wilting of shoots and makes fruits unmarketable. Severe infestations can result in 100% yield loss.",
               "Not a known vector for viruses.", ["Eggplant"],
               "Practice good crop sanitation, use pheromone traps, apply insecticides judiciously, and remove infested shoots.")
]

# 🔎 Unified Pest Search Function
def get_insect_by_name(name):
    """Search for an insect pest by name."""
    return next((pest for pest in insect_pests if pest.name.lower() == name.lower()), None)

# 📢 Example usage: Interactive search
if __name__ == "__main__":
    insect_name = input("🔍 Enter the name of the pest to search for: ")
    result = get_insect_by_name(insect_name)

    if result:
        print("\n🦟 Pest found:\n")
        print(result)
    else:
        print("❌ No pest found under this name.")
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
        attributes = vars(self)
        return "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in attributes.items())

# 📌 Unified List of Insect Pests
insect_pests = [
    InsectPest("Aphid", "Macrosiphum euphorbiae, Myzus persicae", "Small, pear-shaped insects that reproduce rapidly.",
               "Causes chlorosis, leaf curling, and flower abscission.",
               "Vectors viruses such as Cucumber mosaic virus and Potato leaf roll virus.",
               ["Peppers", "Eggplants"], "Use insecticidal soaps, natural predators like ladybugs, and crop rotation."),

    InsectPest("Eggplant Fruit and Shoot Borer", "Leucinodes orbonalis",
               "A major pest of eggplant in South and Southeast Asia. Larvae bore into young shoots and fruits.",
               "Causes wilting of shoots and makes fruits unmarketable. Severe infestations can result in 100% yield loss.",
               "Not a known vector for viruses.", ["Eggplant"],
               "Practice good crop sanitation, use pheromone traps, apply insecticides judiciously, and remove infested shoots.")
]

# 🔎 Pest Search Function
def get_insect_by_name(name):
    """Search for an insect pest by name."""
    return next((pest for pest in insect_pests if pest.name.lower() == name.lower()), None)

# 📌 Class for Nematode Diseases
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

# 📌 Unified List of Nematode Diseases
nematode_diseases = [
    NematodeDisease(
        "Root-Knot Nematodes",
        ["Meloidogyne incognita", "Meloidogyne javanica"],
        "Worldwide",
        "Causes bead-like galls on roots, stunted growth, nutrient deficiency, wilting.",
        "Thrives in warm climates with sandy soil.",
        "Fumigation, crop rotation with non-hosts, resistant plant varieties."
    )
]

# 🔎 Nematode Disease Search Function
def get_nematode_disease_by_name(name):
    """Search for a nematode disease by name."""
    return next((disease for disease in nematode_diseases if disease.name.lower() == name.lower()), None)

# 📢 Example usage
if __name__ == "__main__":
    insect_name = input("🔍 Enter the name of the pest to search for: ")
    result = get_insect_by_name(insect_name)
    if result:
        print("\n🦟 Pest found:\n", result)
    else:
        print("❌ No pest found under this name.")
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
    return next((pest for pest in insect_pests if pest.name.lower() == name.lower()), "❌ Pest not found.")

# 📌 Class for Nematode Diseases
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

print("Exécution terminée avec succès !")
