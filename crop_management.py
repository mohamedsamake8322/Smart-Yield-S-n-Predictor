import disease_info
import nematode_diseases
import insect_pests
import parasitic_plants
import torch
from PIL import Image
import streamlit as st
from disease_model import load_disease_model, predict_disease
from report_generator import generate_pdf_report  
from disease_info import search_disease
from insect_pests import search_pest, predict_pest_control

# Chargement du modèle de reconnaissance des maladies
MODEL_PATH = "disease_model.pth"
try:
    disease_model = load_disease_model(MODEL_PATH)
    print("✅ Disease detection model loaded successfully!")
except Exception as e:
    disease_model = None
    print(f"🛑 Error loading disease detection model: {e}")

def main():
    print("🌱 Welcome to the pest and disease management application! 🌍")

def search_disease():
    disease_name = input("🔍 Enter the name of the disease to search for: ")
    disease = disease_info.get_disease_by_name(disease_name) or nematode_diseases.get_nematode_disease_by_name(disease_name) or parasitic_plants.get_parasitic_plant_by_name(disease_name)
    print(f"\n🦠 Disease found: {disease}" if disease else "❌ No disease found under this name.")

def search_pest():
    insect_name = input("🔍 Enter the name of the pest to search for: ")
    insect = insect_pests.get_insect_by_name(insect_name)
    print(f"\n🦟 Pest found: {insect}" if insect else "❌ No pest found under this name.")

def predict_pest_control():
    insect_name = input("🔍 Enter the name of the pest for control prediction: ")
    climate = input("🌡️ Enter the climate (e.g., hot, humid, dry): ")
    soil = input("🌱 Enter soil type (e.g., sandy, clay, loamy): ")
    crop_stage = input("🌾 Enter crop stage (young plants, growing, mature): ")
    prediction = insect_pests.predict_pest_control(insect_name, climate, soil, crop_stage)
    print(f"\n🤖 Pest control recommendations:\n{prediction}")

def detect_disease_via_image():
    """Détection de maladie végétale via image."""
    st.subheader("🦠 Plant Disease Detection")
    image_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
    
    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Leaf Image", use_column_width=True)

        if st.button("🔍 Detect Disease"):
            if not disease_model:
                st.error("🛑 Disease detection model is not loaded.")
                return

            label = predict_disease(disease_model, image)
            detected_plant = label.split()[0] if label else "Unknown"
            st.success(f"✅ Disease Detection Result: **{label}**")
            st.info(f"🪴 Detected Plant: **{detected_plant}**")

            if "healthy" in label.lower():
                st.success("✅ This leaf appears healthy.")
                st.markdown("👨‍🌾 Recommendation: Continue regular monitoring and maintain good agricultural practices.")
            else:
                st.error("⚠️ Disease detected!")
                st.markdown(
                    """
                    <div style='background-color:#fff3cd;padding:10px;border-left:5px solid #f0ad4e;border-radius:5px'>
                    <b>👩‍⚕️ Suggested Advice:</b>
                    <ul>
                        <li>Isolate the infected plant if possible</li>
                        <li>Use appropriate fungicides or pesticides</li>
                        <li>Improve soil drainage and avoid overwatering</li>
                        <li>Consult an agronomist for accurate diagnosis and treatment</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True
                )

            if st.checkbox("📄 Generate PDF Report"):
                report_pdf = generate_pdf_report(
                    "User",
                    features={"Detected Plant": detected_plant, "Detected Disease": label},
                    prediction="N/A",
                    recommendation="Follow treatment guidelines and monitor the plant closely."
                )
                st.download_button("📥 Download Disease Report", report_pdf, "disease_report.pdf")

# 🔹 Déclaration correcte de `choices`, avant la boucle `while True`
choices = {
    "1": search_disease,
    "2": search_pest,
    "3": nematode_diseases.add_nematode_disease,
    "4": predict_pest_control,
    "5": detect_disease_via_image,
    "6": lambda: print("👋 Goodbye and happy crop management! 🌾")  
}

# 🔹 Suppression de la double boucle `while True`
while True:
    print("\nChoose an option:")
    for num, action in choices.items():
        print(f"{num}️⃣ {action.__name__.replace('_', ' ').capitalize()}")

    choice = input("👉 Enter the number of your choice: ")

    func = choices.get(choice)
    if func:
        func()
        if choice == "6":
            break  
    else:
        print("⚠️ Invalid option, please try again.")

if __name__ == "__main__":
    main()

print("Exécution terminée avec succès !")
