from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

def generate_pdf_report(user_name, features, recommendation):
    """
    Generates a PDF report with detected disease details and recommendations.
    
    Args:
        user_name (str): Name of the user.
        features (dict): Dictionary containing detected disease details.
        recommendation (str): Suggested management approach.

    Returns:
        str: Path of the generated PDF file.
    """
    file_path = f"{user_name}_plant_disease_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    
    styles = getSampleStyleSheet()
    content = []
    
    # Title
    title = Paragraph(f"<b>🌱 Plant Disease Report</b>", styles["Title"])
    content.append(title)
    content.append(Spacer(1, 12))
    
    # User name
    user_info = Paragraph(f"<b>Report Generated for:</b> {user_name}", styles["BodyText"])
    content.append(user_info)
    content.append(Spacer(1, 12))
    
    # Disease Details
    for key, value in features.items():
        detail = Paragraph(f"<b>{key}:</b> {value}", styles["BodyText"])
        content.append(detail)
        content.append(Spacer(1, 8))

    # Recommendations
    rec_section = Paragraph("<b>🌿 Disease Management Recommendations:</b>", styles["Heading2"])
    content.append(rec_section)
    content.append(Spacer(1, 6))
    rec_text = Paragraph(recommendation, styles["BodyText"])
    content.append(rec_text)
    
    # Generate PDF
    doc.build(content)
    return file_path

# Example Usage
if __name__ == "__main__":
    user = "Mohamed"
    disease_details = {
        "Detected Plant": "Tomato",
        "Detected Disease": "Blight",
        "Severity": "High",
    }
    advice = "Apply fungicide, remove infected leaves, improve air circulation."
    
    pdf_file = generate_pdf_report(user, disease_details, advice)
    print(f"✅ Report successfully generated: {pdf_file}")
print("Exécution terminée avec succès !")
