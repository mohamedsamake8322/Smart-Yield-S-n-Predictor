class AbioticDisease:
    def __init__(self, name, causal_agents, affected_crops, distribution, symptoms, conditions, control):
        self.name = name
        self.causal_agents = causal_agents
        self.affected_crops = affected_crops
        self.distribution = distribution
        self.symptoms = symptoms
        self.conditions = conditions
        self.control = control

    def __str__(self):
        details = vars(self)  # Récupère les attributs sous forme de dictionnaire
        return "\n".join(f"{key.capitalize().replace('_', ' ')}: {value}" for key, value in details.items())

# Bibliothèque sous forme de dictionnaire pour un accès plus rapide
abiotic_diseases = {
    "Blossom-End Rot": AbioticDisease(
        "Blossom-End Rot", ["Calcium Imbalance"], ["Tomatoes", "Peppers"], "Worldwide",
        "Water-soaked lesions near blossom scar, turns leathery brown, colonized by saprophytic fungi.",
        "Insufficient calcium uptake, alternating wet/dry soil, root stress, excess nitrogen.",
        "Drip irrigation, lime for calcium, avoid ammonium fertilizers, apply calcium nitrate."
    ),
    "Chemical Damage": AbioticDisease(
        "Chemical Damage", ["Herbicides", "Insecticides"], ["Peppers", "Eggplants", "Tomatoes"], "Worldwide",
        "Chlorosis, necrotic spots, leaf twisting, stem swelling/cracking, malformed fruit.",
        "Excessive chemical use, drift from nearby fields, improper application/weather.",
        "Apply chemicals correctly, avoid contamination, proper storage and cleaning."
    ),
    "Chimera": AbioticDisease(
        "Chimera", ["Genetic Mutation"], ["Peppers", "Eggplants", "Tomatoes"], "Worldwide",
        "Variegated leaves, chlorophyll loss, filiform leaves, distorted fruit and growth points.",
        "Spontaneous genetic mutations.",
        "Use high-quality seeds to minimize occurrence."
    ),
    "Cracking": AbioticDisease(
        "Cracking", ["Environmental", "Genetic"], ["Peppers", "Tomatoes"], "Worldwide",
        "Superficial or deep cracks on fruit, secondary infections causing postharvest decay.",
        "Rapid fruit growth, stress, fluctuating temperatures or humidity, heavy rain.",
        "Optimized irrigation and nutrition, avoiding high humidity in greenhouses."
    )
}

def get_abiotic_disease_by_name(name):
    """Recherche rapide d'une maladie abiotique."""
    return abiotic_diseases.get(name, None)

# Exemple d'affichage
print(get_abiotic_disease_by_name("Blossom-End Rot"))
print("Exécution terminée avec succès !")
