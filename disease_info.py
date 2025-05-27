class Disease:
    def __init__(self, name, causal_agents, distribution, symptoms, conditions, control, vectors=None):
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
        name="Anthracnose",
        causal_agents=["Colletotrichum capsici", "C. gloeosporioides", "C. coccodes", "C. acutatum"],
        distribution="Worldwide",
        symptoms="Affects all above-ground parts of peppers. Fruit lesions are the most economically important. "
                  "Under moist conditions, pink, salmon, or orange masses of spores are formed.",
        conditions="Warm, wet weather favors infection. Optimal temperatures range from 20° to 27°C.",
        control="Use high-quality seeds, crop rotation, remove weeds and infected debris, minimize fruit wounds."
    ),
    "Cercospora Leaf Spot": Disease(
        name="Cercospora Leaf Spot (Frogeye)",
        causal_agents=["Cercospora capsici", "C. melongenae"],
        distribution="Worldwide",
        symptoms="Affects leaves, petioles, stems, and peduncles of pepper and eggplant. "
                  "Lesions appear chlorotic, later turn necrotic with light-gray centers and dark-brown margins. "
                  "Concentric rings form as lesions expand, resembling frog eyes.",
        conditions="Fungi survive in plant debris for at least one year. Wet, warm weather favors development.",
        control="Protective fungicide spray program with cultural practices reduces losses."
    ),
    "Choanephora Blight": Disease(
        name="Choanephora Blight (Wet Rot)",
        causal_agents=["Choanephora cucurbitarum"],
        distribution="Worldwide in tropical regions",
        symptoms="Visible on apical growing points, flowers, and fruits. "
                  "Initially, water-soaked areas develop on leaves, and apical growing points become blighted. "
                  "Dark-gray fungal growth appears, with silvery spine-like structures and dark spores.",
        conditions="Found throughout tropical areas on many crops. High humidity and rain favor disease development.",
        control="Few management techniques available; fungicide sprays may help reduce damage."
    )
}

# Efficient disease retrieval
def get_disease_info(name):
    """🔎 Searches for a disease by its name in the database."""
    return DISEASE_DATABASE.get(name, "⚠️ Disease not found.")
class Disease:
    def __init__(self, name, causal_agents, distribution, symptoms, conditions, control, vectors=None):
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

# 📌 Optimized disease database
DISEASE_DATABASE = {
    "Fusarium Wilt": Disease(
        "Fusarium Wilt",
        ["Fusarium oxysporum f. sp. capsici (Pepper)", "Fusarium oxysporum f. sp. melongenae (Eggplant)"],
        "Pepper: Argentina, Italy, Mexico, USA | Eggplant: Israel, Italy, Japan, Netherlands, USA",
        "Early symptoms include slight yellowing of foliage and wilting of upper leaves. "
        "Leaves may turn dull-green to brown and remain attached. Cutting stems reveals reddish-brown streaks.",
        "The fungus survives in soil for years, spreading via farm equipment, irrigation water, and infected debris. "
        "Warm soil temperatures (33°C/92°F) and high moisture accelerate disease development.",
        "Plant on raised beds to improve drainage. Thoroughly disinfect equipment before moving from infested fields."
    ),
    "Gray Leaf Spot": Disease(
        "Gray Leaf Spot",
        ["Stemphylium solani", "Stemphylium lycopersici"],
        "Worldwide",
        "Small red-to-brown spots develop on leaves, petioles, stems, and peduncles. "
        "Lesions expand into white-to-gray centers with red-brown margins. Severe infections cause leaf yellowing and drop.",
        "Fungi survive year-round in soil and plant debris. Spores spread via wind and splashing water under warm, humid conditions.",
        "Remove plant debris, ensure ventilation for seedlings, and apply fungicides to reduce losses."
    ),
    "White Mold (Pink Rot, Watery Soft Rot)": Disease(
        "White Mold (Pink Rot, Watery Soft Rot)",
        ["Sclerotinia sclerotiorum"],
        "Worldwide",
        "Dark-green, water-soaked lesions develop on foliage, stems, and fruit. Stem infections girdle the plant at the soil line, causing wilting. "
        "White cotton-like mycelium blankets affected tissue, forming black sclerotia as the disease progresses.",
        "The fungus survives as sclerotia in soil and plant debris. Favored by dew, fog, frequent rain, and temperate conditions. "
        "Long-distance spread occurs via airborne spores; irrigation water and contaminated soil contribute locally.",
        "Plant in well-drained soil, use wide row spacing, remove plant debris, and rotate crops with non-host plants like corn and grasses."
    ),
    "Cucumber Mosaic Virus": Disease(
        "Cucumber Mosaic Virus",
        ["Cucumber mosaic virus"],
        "Worldwide",
        "Stunted growth, leaf mottling, and distorted fruit development.",
        "Spreads rapidly under warm conditions.",
        "Use virus-free seeds, remove infected plants, and control aphid populations.",
        ["Aphid"]
    ),
    "Beet Curly Top Virus": Disease(
        "Beet Curly Top Virus",
        ["Beet curly top virus"],
        "Worldwide",
        "Stunted plants, leaf curling, and yellowing.",
        "Favored by dry and warm climates.",
        "Implement vector control and remove infected plants.",
        ["Beet Leafhopper (Circulifer tenellus)"]
    ),
    "Little Leaf Disease": Disease(
        "Little Leaf Disease",
        ["Phytoplasma spp."],
        "Asia",
        "Small, yellow patches on leaves, severe interveinal yellowing, necrotic areas.",
        "Triggered by leafhopper infestations.",
        "Use row covers, remove weeds, and control leafhopper populations.",
        ["Cotton Leafhopper (Hishimonus phycitis)"]
    )
}

# Efficient disease retrieval
def get_disease_info(name):
    """🔎 Searches for a disease by its name in the database."""
    return DISEASE_DATABASE.get(name, "⚠️ Disease not found.")
print("Exécution terminée avec succès !")
