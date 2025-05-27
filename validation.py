#validation
def validate_input(crop, pH, soil_type, growth_stage, temperature, humidity):
    """ Vérifie que les entrées utilisateur sont valides. """
    if not crop or not soil_type or not growth_stage:
        return False, "🚨 Missing crop, soil type, or growth stage!"
    
    if pH < 3.5 or pH > 9.0:
        return False, "🚨 pH out of acceptable range (3.5 - 9.0)!"
    
    if temperature < -10 or temperature > 50:
        return False, "🚨 Temperature seems unrealistic!"
    
    if humidity < 0 or humidity > 100:
        return False, "🚨 Humidity percentage must be between 0-100!"
    
    return True, None
