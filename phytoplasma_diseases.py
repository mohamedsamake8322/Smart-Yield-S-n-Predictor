import random
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

MODEL_PATH = "model/phytoplasma_model.pkl"

# 📌 Definition of phytoplasma diseases with new characteristics
phytoplasma_diseases = [
    {"name": "Little Leaf Phytoplasma", "symptoms": "Small pale-green leaves, shortened stems", "climate": "warm/humid", "soil_type": "loamy", "leaves": "yellowing", "stems": "shortened", "fruits": "sterile", "roots": "weak"},
    {"name": "Stolbur Phytoplasma", "symptoms": "Stunting, chlorosis, abnormal flowering", "climate": "warm/dry", "soil_type": "sandy", "leaves": "chlorotic", "stems": "stunted", "fruits": "deformed", "roots": "underdeveloped"},
    {"name": "Aster Yellows Phytoplasma", "symptoms": "Leaf deformation, sterile flowering", "climate": "humid", "soil_type": "clay", "leaves": "deformed", "stems": "weak", "fruits": "sterile", "roots": "damaged"},
    {"name": "Papaya Bunchy Top Phytoplasma", "symptoms": "Stunted growth, deformed leaves", "climate": "hot/humid", "soil_type": "sandy", "leaves": "deformed", "stems": "stunted", "fruits": "small", "roots": "rotting"},
]

# ✅ Data preparation
df = pd.DataFrame(phytoplasma_diseases)
df["label"] = LabelEncoder().fit_transform(df["name"])  # Convert diseases to numerical values

# 🔄 Encoding categorical variables
label_encoders = {col: LabelEncoder() for col in ["symptoms", "climate", "soil_type", "leaves", "stems", "fruits", "roots"]}
for col in label_encoders:
    df[col] = label_encoders[col].fit_transform(df[col])  # Convert text to numbers

# 🔄 Splitting the data into training and testing sets
X = df[["symptoms", "climate", "soil_type", "leaves", "stems", "fruits", "roots"]]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 🚀 Model training
classifier = RandomForestClassifier(n_estimators=200, random_state=42)
classifier.fit(X_train, y_train)

# 💾 Saving the model
os.makedirs("model", exist_ok=True)
joblib.dump((classifier, label_encoders), MODEL_PATH)
print(f"✅ Phytoplasma model trained and saved under {MODEL_PATH}!")

# 🔎 Prediction function with improved symptom recognition
def predict_phytoplasma_disease(symptom, climate, soil_type, leaves, stems, fruits, roots):
    """Predicts the phytoplasma disease based on symptoms, climate, soil type, and leaf, stem, fruit, and root characteristics."""
    model, label_encoders = joblib.load(MODEL_PATH)

    # 🚨 Find the closest match for the symptoms
    best_match = None
    for known_symptom in label_encoders["symptoms"].classes_:
        if symptom.lower() in known_symptom.lower():
            best_match = known_symptom
            break

    if best_match is None:
        return {"error": f"❌ Symptom '{symptom}' not recognized. Try: {label_encoders['symptoms'].classes_.tolist()}"}

    # 🔄 Transform user inputs into numerical values
    features = pd.DataFrame([[best_match, climate, soil_type, leaves, stems, fruits, roots]], columns=["symptoms", "climate", "soil_type", "leaves", "stems", "fruits", "roots"])
    for col in label_encoders:
        features[col] = label_encoders[col].transform(features[col])  

    prediction = model.predict(features)[0]
    disease_name = df.loc[df["label"] == prediction, "name"].values[0]

    return {"Predicted Disease": disease_name}

# 🔥 Prediction test with leaves, stems, fruits, and roots
print(predict_phytoplasma_disease("Leaf deformation", "humid", "clay", "deformed", "weak", "sterile", "damaged"))
