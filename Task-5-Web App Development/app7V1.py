#cliicalbert et bioGPT
#use with artifcts from bioGPT_pipeline.py

#madakixo

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel, BioGptTokenizer, BioGptForCausalLM
import pickle
import plotly.express as px

# Configure Streamlit page
st.set_page_config(
    page_title="Disease Diagnosis & Treatment Generator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Define paths (adjust these based on your setup)
save_folder = "./model_classifier-clinicalbert-biogpt"  # Local path or adjust for Google Drive
biogpt_folder = "microsoft/BioGPT"  # Use pre-trained BioGPT; update to fine-tuned path if available
data_path = "./processed_diseases-priority.csv"  # Local path or adjust for Google Drive

# ClinicalBERT Model Definition
class ClinicalBERTDiseaseClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=1):  # num_classes set dynamically
        super(ClinicalBERTDiseaseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return Quinta

# Load models and artifacts
@st.cache_resource
def load_models_and_artifacts(save_folder, biogpt_folder, num_classes):
    """Load ClinicalBERT model, label encoder, and BioGPT."""
    # Load ClinicalBERT components
    clinicalbert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    clinicalbert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    clinicalbert_model.eval()

    model = ClinicalBERTDiseaseClassifier(input_dim=768, num_classes=num_classes).to(device)
    checkpoint = torch.load(os.path.join(save_folder, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(os.path.join(save_folder, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    # Load BioGPT
    biogpt_tokenizer = BioGptTokenizer.from_pretrained(biogpt_folder)
    biogpt_model = BioGptForCausalLM.from_pretrained(biogpt_folder).to(device)
    biogpt_model.eval()

    return clinicalbert_tokenizer, clinicalbert_model, model, label_encoder, biogpt_tokenizer, biogpt_model

# Load symptoms from dataset
@st.cache_data
def load_symptoms(data_path):
    """Extract unique symptoms from the dataset."""
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    all_symptoms = set()
    for symptoms in df["Symptoms"]:
        all_symptoms.update([s.strip() for s in symptoms.split(",")])
    return sorted(list(all_symptoms))

# Prediction function
def predict(symptoms, clinicalbert_tokenizer, clinicalbert_model, model, label_encoder, biogpt_model, biogpt_tokenizer, confidence_threshold=0.5):
    """Predict disease with ClinicalBERT and generate treatment with BioGPT."""
    # Disease prediction with ClinicalBERT
    with torch.no_grad():
        inputs = clinicalbert_tokenizer(symptoms, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
        outputs = clinicalbert_model(**inputs)
        symptoms_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    symptoms_tensor = symptoms_embedding.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(symptoms_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    if confidence < confidence_threshold:
        return "Uncertain prediction", "Unable to suggest treatment due to low confidence", confidence

    disease_name = label_encoder.inverse_transform([predicted_class])[0]

    # Treatment generation with BioGPT
    prompt = f"Given the symptoms '{symptoms}' and the diagnosed disease '{disease_name}', suggest a treatment plan."
    input_ids = biogpt_tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = biogpt_model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    treatment = biogpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return disease_name, treatment, confidence

# Main application
def main():
    # Load dataset for num_classes and symptoms
    df = pd.read_csv(data_path)
    num_classes = len(df["Disease"].unique())

    # Load models
    try:
        clinicalbert_tokenizer, clinicalbert_model, model, label_encoder, biogpt_tokenizer, biogpt_model = load_models_and_artifacts(
            save_folder, biogpt_folder, num_classes
        )
        common_symptoms = load_symptoms(data_path)
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Sidebar
    st.sidebar.title("🩺 Symptom Analyzer")
    st.sidebar.markdown("Input symptoms to diagnose diseases and generate treatments.")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920336.png", width=100)

    with st.sidebar.expander("Settings", expanded=False):
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=1.0, value=0.5, step=0.05,
            help="Minimum confidence for a valid prediction."
        )

    # Main content
    st.title("Disease Diagnosis & Treatment Generator")
    st.markdown("**Enter symptoms to predict diseases and receive AI-generated treatment plans!**")

    # Symptom input
    st.subheader("Symptoms")
    selected_symptoms = st.multiselect(
        "Select common symptoms",
        options=common_symptoms,
        default=["Fever"],
        help="Choose from the list or add custom symptoms below."
    )

    manual_symptoms = st.text_area(
        "Custom Symptoms (comma-separated)",
        placeholder="e.g., cough, fatigue",
        height=100,
        help="Add additional symptoms here."
    )

    symptoms = ", ".join(selected_symptoms + [s.strip() for s in manual_symptoms.split(",") if s.strip()])
    st.write(f"**Symptoms Entered:** {symptoms if symptoms else 'None'}")

    if symptoms:
        st.info("Ready to predict!")
    else:
        st.warning("Please enter at least one symptom.")

    # Predict button
    if st.button("Generate Prediction", key="predict_button"):
        if symptoms:
            progress_bar = st.progress(0)
            with st.spinner("Processing..."):
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                try:
                    disease, treatment, confidence = predict(
                        symptoms, clinicalbert_tokenizer, clinicalbert_model, model, label_encoder,
                        biogpt_model, biogpt_tokenizer, confidence_threshold
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Diagnosis")
                        badge_color = "green" if confidence >= 0.75 else "orange" if confidence >= 0.5 else "red"
                        st.markdown(
                            f"**Disease:** {disease}  <span style='background-color:{badge_color};color:white;padding:2px 5px;border-radius:3px'>{confidence:.2%}</span>",
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.subheader("Treatment Plan")
                        st.write(f"**Suggested Treatment:** {treatment}")

                    # Confidence chart
                    st.subheader("Confidence Level")
                    fig = px.bar(
                        x=["Predicted Disease"], y=[confidence],
                        labels={"x": "Prediction", "y": "Confidence"},
                        title="Prediction Confidence",
                        color=[confidence], color_continuous_scale="Blues",
                        range_y=[0, 1]
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Store prediction in session state
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "Symptoms": symptoms,
                        "Disease": disease,
                        "Treatment": treatment,
                        "Confidence": f"{confidence:.2%}"
                    })

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            progress_bar.empty()
        else:
            st.error("No symptoms provided!")

    # Display prediction history
    if "history" in st.session_state and st.session_state.history:
        st.subheader("Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Disclaimer:** This app uses ClinicalBERT for disease prediction and BioGPT for treatment generation.
        Results are AI-generated and should be reviewed by a healthcare professional before use.
        Powered by Streamlit, PyTorch, and Transformers.
        """
    )

if __name__ == "__main__":
    main()
