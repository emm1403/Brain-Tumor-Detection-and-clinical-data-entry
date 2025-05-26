import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from textwrap import wrap
import os
import requests
import gdown

st.set_page_config(page_title="Brain Tumor Detection", page_icon=None)
st.title("Brain Tumor Detection and Clinical Data Entry")
st.markdown("Please fill in the clinical data first. Then upload MRI image(s) and click 'Start Prediction'.")

def download_model_gdown(google_drive_id, destination):
    url = f"https://drive.google.com/uc?id={google_drive_id}"
    gdown.download(url, destination, quiet=True)
    # st.success("Modelo descargado correctamente.")  # Comentado para no mostrar

model_path = "best_model.keras"
google_drive_id = "1KUqfzzkVsBL1pYf5OizRFmJz90RjzaQc"

if not os.path.exists(model_path):
    # st.info("Descargando modelo desde Google Drive con gdown...")  # Comentado para no mostrar
    download_model_gdown(google_drive_id, model_path)

# st.write("Tamaño del modelo:", os.path.getsize(model_path))  # Comentado para no mostrar
# st.write("Ruta completa del modelo:", os.path.abspath(model_path))  # Comentado para no mostrar
# st.write("¿El archivo existe?", os.path.exists(model_path))  # Comentado para no mostrar

if os.path.exists(model_path):
    # st.success(f"Modelo encontrado.")  # Comentado para no mostrar
    try:
        model = tf.keras.models.load_model(model_path)
        # st.success("Modelo cargado correctamente.")  # Comentado para no mostrar
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
else:
    st.error(f"No se encontró el archivo del modelo en: {model_path}")


class_names = ['Glioma Tumour', 'Meningioma Tumour', 'No Tumour', 'Pituitary Tumour']

recommendations_dict = {
    "Glioma Tumour": (
        "Recommendation: Refer to a neuro-oncologist for further imaging and biopsy confirmation. "
        "Treatment often involves surgery followed by radiotherapy and chemotherapy depending on the grade."
    ),
    "Meningioma Tumour": (
        "Recommendation: Evaluate size and symptoms. Small, asymptomatic meningiomas may be observed, "
        "while symptomatic or large tumors usually require surgical resection."
    ),
    "No Tumour": (
        "Recommendation: No tumor detected. Consider alternative diagnoses and clinical follow-up as needed."
    ),
    "Pituitary Tumour": (
        "Recommendation: Endocrinological evaluation is essential. Treatment options include surgery, "
        "medical therapy, or radiotherapy based on tumor size and hormone secretion."
    ),
}

st.sidebar.header("Professional and Patient Information")
professional_name = st.sidebar.text_input("Attending Professional's Name")
patient_name = st.sidebar.text_input("Patient's Name")

st.sidebar.header("Select Patient Clinical Features")
clinical_features = {}
clinical_features["Age Range"] = st.sidebar.selectbox("Age Range:", ["None", "Under 20", "20-29", "30-39", "40-49", "50-59", "60-69", "70 or older"])
clinical_features["Symptom Onset"] = st.sidebar.selectbox("Symptom Onset:", ["None", "Acute", "Subacute", "Progressive", "Insidious", "Other"])
clinical_features["Focal Neurological Symptoms"] = st.sidebar.selectbox("Focal Neurological Symptoms:", ["None", "Motor deficits", "Aphasia", "Sensory deficits", "Visual field deficit", "Other"])
clinical_features["Seizures"] = st.sidebar.selectbox("Seizures:", ["None", "Frequent", "Occasional", "Rare"])
clinical_features["Visual Symptoms"] = st.sidebar.selectbox("Visual Symptoms:", ["None", "Blurred vision", "Hemianopsia", "Double vision", "Other"])
clinical_features["Endocrine Symptoms"] = st.sidebar.selectbox("Endocrine Symptoms:", ["None", "Amenorrhea", "Galactorrhea", "Hypothyroidism", "Other"])
clinical_features["Intracranial Hypertension"] = st.sidebar.selectbox("Intracranial Hypertension (headache, nausea):", ["None", "Mild", "Moderate", "Severe"])
clinical_features["Personality/Cognitive Change"] = st.sidebar.selectbox("Personality or Cognitive Changes:", ["None", "Mild", "Moderate", "Severe"])
clinical_features["Family History of Tumors"] = st.sidebar.selectbox("Family History of Tumors:", ["None", "Positive", "Unknown"])
clinical_features["Imaging Findings"] = st.sidebar.selectbox("Imaging Findings (MRI/CT):", ["None", "Infiltrative lesion", "Well-defined lesion", "Sella turcica lesion", "Normal or non-tumoral"])

uploaded_files = st.file_uploader("Upload MRI image(s) (only after filling clinical data)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Start Prediction"):
    if not uploaded_files:
        st.warning("Please upload at least one MRI image to start prediction.")
    else:
        for uploaded_file in uploaded_files:
            st.markdown("---")
            st.subheader(f"Image: {uploaded_file.name}")

            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

            img = np.array(image)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            with st.spinner('Analyzing...'):
                preds = model.predict(img)
                predicted_class = class_names[np.argmax(preds)]
                confidence = np.max(preds) * 100

                st.success(f"Predicted Tumor Type: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}%")

                if confidence < 90:
                    st.warning("Low confidence. Please try uploading a higher quality image to reduce diagnostic errors.")

                st.markdown("### Prediction Probabilities:")
                for i, prob in enumerate(preds[0]):
                    st.write(f"- {class_names[i]}: {prob * 100:.2f}%")

                rec = recommendations_dict.get(predicted_class, "No recommendations available.")
                st.markdown(f"### Clinical Recommendations\n{rec}")

                st.markdown("### Patient & Professional Info")
                st.write(f"- Professional: {professional_name if professional_name else 'N/A'}")
                st.write(f"- Patient: {patient_name if patient_name else 'N/A'}")

                st.markdown("### Clinical Data Provided")
                for key, value in clinical_features.items():
                    st.write(f"- **{key}:** {value}")

                def generate_pdf(image_name, prediction, confidence, professional, patient, clinical_data, recommendations):
                    buffer = BytesIO()
                    c = canvas.Canvas(buffer, pagesize=letter)
                    width, height = letter
                    margin_x = 50
                    y = height - 50

                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(margin_x, y, "Brain Tumor Detection Report")
                    y -= 25

                    c.setFont("Helvetica", 12)
                    c.drawString(margin_x, y, f"Professional: {professional if professional else 'N/A'}")
                    y -= 20
                    c.drawString(margin_x, y, f"Patient: {patient if patient else 'N/A'}")
                    y -= 20
                    c.drawString(margin_x, y, f"Image: {image_name}")
                    y -= 20
                    c.drawString(margin_x, y, f"Predicted Tumor Type: {prediction}")
                    y -= 20
                    c.drawString(margin_x, y, f"Confidence Level: {confidence:.2f}%")
                    y -= 20

                    if confidence < 90:
                        c.setFillColorRGB(1, 0, 0)
                        c.drawString(margin_x, y, "Warning: Low confidence. Please consider uploading a clearer image.")
                        c.setFillColorRGB(0, 0, 0)
                        y -= 20

                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(margin_x, y, "Clinical Data:")
                    y -= 20

                    c.setFont("Helvetica", 12)
                    for key, value in clinical_data.items():
                        if value and value != "None":
                            wrapped_lines = wrap(f"- {key}: {value}", width=85)
                            for line in wrapped_lines:
                                if y < 50:
                                    c.showPage()
                                    y = height - 50
                                    c.setFont("Helvetica", 12)
                                c.drawString(margin_x + 5, y, line)
                                y -= 15
                    y -= 10

                    c.setFont("Helvetica-Bold", 14)
                    if y < 60:
                        c.showPage()
                        y = height - 50
                    c.drawString(margin_x, y, "Clinical Recommendations:")
                    y -= 20

                    c.setFont("Helvetica", 12)
                    for paragraph in wrap(recommendations, width=100):
                        if y < 50:
                            c.showPage()
                            y = height - 50
                            c.setFont("Helvetica", 12)
                        c.drawString(margin_x + 5, y, paragraph)
                        y -= 15

                    c.showPage()
                    c.save()
                    buffer.seek(0)
                    return buffer

                pdf = generate_pdf(
                    uploaded_file.name,
                    predicted_class,
                    confidence,
                    professional_name,
                    patient_name,
                    clinical_features,
                    rec
                )

                filename_safe = patient_name.replace(" ", "_") if patient_name else "patient"
                pdf_filename = f"report_{filename_safe}_{uploaded_file.name}.pdf"

                st.download_button(
                    label="Download PDF Report",
                    data=pdf,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )
