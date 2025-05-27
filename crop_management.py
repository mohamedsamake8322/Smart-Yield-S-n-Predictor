import disease_info
import nematode_diseases
import insect_pests
import parasitic_plants
import torch
from PIL import Image
import streamlit as st
from disease_model import load_disease_model, predict_disease
from report_generator import generate_pdf_report  # Ajout du module pour générer des rapports

# Chargement du modèle de reconnaissance des maladies
MODEL_PATH = "resnet18_disease_model.pth"
try:
    disease_model = load_disease_model(MODEL_PATH)
except Exception as e:
    disease_model = None
    print(f"🛑 Error loading disease detection model: {e}")

def main():
    print("🌱 Welcome to the pest and disease management application! 🌍")

    while True:
        print("\nChoose an option:")
        print("1️⃣ Search for a disease")
        print("2️⃣ Search for a pest")
        print("3️⃣ Add a nematode disease")
        print("4️⃣ Get a pest control prediction")
        print("5️⃣ Detect plant disease via image")
        print("6️⃣ Exit")

        choice = input("👉 Enter the number of your choice: ")

        if choice == "1":
            disease_name = input("🔍 Enter the name of the disease to search for: ")
            disease = disease_info.get_disease_by_name(disease_name) or nematode_diseases.get_nematode_disease_by_name(disease_name) or parasitic_plants.get_parasitic_plant_by_name(disease_name)

            if disease:
                print("\n🦠 Disease found:")
                print(disease)
            else:
                print("❌ No disease found under this name.")

        elif choice == "2":
            insect_name = input("🔍 Enter the name of the pest to search for: ")
            insect = insect_pests.get_insect_by_name(insect_name)

            if insect:
                print("\n🦟 Pest found:")
                print(insect)
            else:
                print("❌ No pest found under this name.")

        elif choice == "3":
            print("\n➕ Add a new nematode disease")
            nematode_diseases.add_nematode_disease()

        elif choice == "4":
            insect_name = input("🔍 Enter the name of the pest for control prediction: ")
            climate = input("🌡️ Enter the climate (e.g., hot, humid, dry): ")
            soil = input("🌱 Enter soil type (e.g., sandy, clay, loamy): ")
            crop_stage = input("🌾 Enter crop stage (young plants, growing, mature): ")

            prediction = insect_pests.predict_pest_control(insect_name, climate, soil, crop_stage)
            print("\n🤖 Pest control recommendations:")
            print(prediction)

        elif choice == "5":
            detect_disease_via_image()

        elif choice == "6":
            print("👋 Goodbye and happy crop management! 🌾")
            break

        else:
            print("⚠️ Invalid option, please try again.")

def detect_disease_via_image():
    """Détection de maladie végétale via image."""
    st.subheader("🦠 Plant Disease Detection")
    image_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
    
    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Leaf Image", use_column_width=True)
        
        if st.button("🔍 Detect Disease"):
            if disease_model:
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
            else:
                st.error("🛑 Disease detection model is not loaded.")

if __name__ == "__main__":
    main()
