import random

class DiseaseRiskPredictor:
    def __init__(self, disease_name, temperature, humidity, wind_speed, soil_type, aphid_population, crop_stage, season):
        self.disease_name = disease_name
        self.temperature = temperature
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.soil_type = soil_type
        self.aphid_population = aphid_population
        self.crop_stage = crop_stage
        self.season = season

    def get_seasonal_adjustment(self):
        """Adjusts risk based on the season."""
        season_factors = {
            "spring": 0.1,  
            "summer": 0.2,  
            "autumn": 0.15,  
            "winter": 0.05  
        }
        return season_factors.get(self.season.lower(), 0)

    def calculate_risk(self):
        """Calculates the infection risk based on environmental conditions."""
        base_risk = random.uniform(0.0, 0.6)  # Adjusted for better accuracy  

        disease_factors = {
            "viral": {"temperature_range": (25, 35), "humidity_range": (50, 80), "insect": "aphid"},
            "bacterial": {"temperature_range": (18, 30), "humidity_range": (70, 100), "insect": None},
            "fungal": {"temperature_range": (10, 25), "humidity_range": (80, 100), "insect": None},
            "phytoplasma": {"temperature_range": (20, 32), "humidity_range": (60, 90), "insect": "leafhopper"},
            "insect_damage": {"temperature_range": (22, 38), "humidity_range": (40, 70), "insect": "thrips"}
        }

        if self.disease_name.lower() in disease_factors:
            factors = disease_factors[self.disease_name.lower()]
            adjustments = [
                (factors["temperature_range"][0] <= self.temperature <= factors["temperature_range"][1], 0.15),
                (factors["humidity_range"][0] <= self.humidity <= factors["humidity_range"][1], 0.20),
                (self.wind_speed > 20 and self.disease_name.lower() == "fungal", 0.25),
                (self.soil_type == "clayey" and self.disease_name.lower() == "bacterial", 0.10),
                (factors["insect"] and self.aphid_population > 500, 0.30)
            ]
            base_risk += sum(value for condition, value in adjustments if condition)

        base_risk += self.get_seasonal_adjustment()  
        base_risk = min(base_risk, 1)  

        print(f"🧐 Evaluating {self.disease_name} risk...\n"
              f"🌡️ Temperature: {self.temperature}°C\n"
              f"💧 Humidity: {self.humidity}%\n"
              f"🍃 Wind Speed: {self.wind_speed} km/h\n"
              f"🌱 Soil Type: {self.soil_type}\n"
              f"🐜 Aphid Population: {self.aphid_population}\n"
              f"🌾 Crop Stage: {self.crop_stage}\n"
              f"🍂 Season: {self.season}\n"
              f"🔍 Final Risk Score: {base_risk:.2f}")

        return base_risk  # 🔹 Correction: The function returns the score

# Example usage of the predictive model
predictor = DiseaseRiskPredictor(
    disease_name="viral",
    temperature=28,
    humidity=65,
    wind_speed=10,
    soil_type="sandy",
    aphid_population=600,
    crop_stage="young plants",
    season="summer"
)

print(predictor.calculate_risk())  # ✅ Displays the score correctly
print("Execution completed successfully!")
