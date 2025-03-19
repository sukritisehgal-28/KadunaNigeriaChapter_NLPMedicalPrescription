#clinicalBERT
#for use with clinicalBert pipeline
#madakixo

import streamlit as st
import pandas as pd  # For data manipulation and DataFrame operations
import numpy as np  # For numerical computations and array handling
import torch  # Core PyTorch library for tensors and neural networks
import torch.nn as nn  # Neural network module for layers and loss
from transformers import AutoTokenizer, AutoModel  # Load ClinicalBERT tokenizer and model
import pickle  # To save/load model artifacts
import os  # For file and directory operations
import time  # For progress bar timing
import plotly.express as px  # For interactive visualizations

# Configure Streamlit page for a clean, professional interface
st.set_page_config(
    page_title="Disease Predictor with ClinicalBERT",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define device for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Define paths (adjust these based on your local or Colab setup)
save_folder = "model_classifier-clinicalbert"  # Local path; update for Colab if needed
data_path = "processed_diseases-priority.csv"  # Local path; update for Colab if needed

# ClinicalBERT Model Definition
class ClinicalBERTDiseaseClassifier(nn.Module):
    """
    Neural network for disease classification using ClinicalBERT embeddings.
    - Input: 768-dimensional embeddings from ClinicalBERT.
    - Architecture: Three-layer feedforward network with batch normalization and dropout.
    - Output: Logits for each disease class.
    """
    def __init__(self, input_dim=768, num_classes=1):
        super(ClinicalBERTDiseaseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),  # Reduce dimensionality from 768 to 512
            nn.BatchNorm1d(512),       # Normalize outputs for stability
            nn.ReLU(),                 # Non-linear activation
            nn.Dropout(0.4),           # Dropout to prevent overfitting
            nn.Linear(512, 256),       # Further reduce to 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # Output layer matches number of disease classes
        )

    def forward(self, x):
        return self.network(x)

# Load models and artifacts with caching for efficiency
@st.cache_resource
def load_models_and_artifacts(save_folder, num_classes):
    """
    Loads ClinicalBERT components and trained artifacts:
    - ClinicalBERT tokenizer and model for embedding extraction.
    - Trained classifier for disease prediction.
    - Label encoder to decode predictions.
    """
    try:
        # Load ClinicalBERT tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        clinicalbert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
        clinicalbert_model.eval()  # Set to evaluation mode

        # Load trained classifier
        model = ClinicalBERTDiseaseClassifier(input_dim=768, num_classes=num_classes).to(device)
        checkpoint = torch.load(os.path.join(save_folder, "best_model.pth"), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load label encoder
        with open(os.path.join(save_folder, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)

        return tokenizer, clinicalbert_model, model, label_encoder
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, None, None

# Load dataset and extract symptoms
@st.cache_data
def load_data_and_symptoms(data_path):
    """
    Loads the dataset and extracts unique symptoms for user selection.
    - Cleans data by removing NaN rows.
    - Returns DataFrame and sorted list of unique symptoms.
    """
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    all_symptoms = set()
    for symptoms in df["Symptoms"]:
        all_symptoms.update([s.strip().lower() for s in symptoms.split(",")])
    return df, sorted(list(all_symptoms))

# Prediction function
def predict(symptoms, tokenizer, clinicalbert_model, model, label_encoder, df, confidence_threshold=0.5):
    """
    Predicts disease and retrieves treatment from the dataset:
    - Extracts ClinicalBERT embeddings for input symptoms.
    - Uses trained classifier to predict disease with confidence.
    - Looks up treatment from the dataset based on predicted disease.
    """
    # Extract ClinicalBERT embedding
    with torch.no_grad():
        inputs = tokenizer(symptoms, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
        outputs = clinicalbert_model(**inputs)
        symptoms_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # Mean pooling

    symptoms_tensor = symptoms_embedding.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(symptoms_tensor)
        probabilities = torch.softmax(output, dim=1)[0]  # Convert logits to probabilities
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    # Check confidence threshold
    if confidence < confidence_threshold:
        return "Uncertain prediction", "Unable to suggest treatment due to low confidence", confidence

    # Decode disease and fetch treatment
    disease_name = label_encoder.inverse_transform([predicted_class])[0]
    disease_info = df[df["Disease"] == disease_name].iloc[0]  # Assumes disease names match original labels
    treatment = disease_info.get("Treatment", "Treatment information not available")

    return disease_name, treatment, confidence

# Main application
def main():
    """
    Main Streamlit application:
    - Loads models, dataset, and symptoms.
    - Provides UI for symptom input and prediction.
    - Displays results with confidence visualization and history.
    """
    # Load dataset and calculate num_classes
    df, common_symptoms = load_data_and_symptoms(data_path)
    num_classes = len(df["Disease"].unique())

    # Load models
    tokenizer, clinicalbert_model, model, label_encoder = load_models_and_artifacts(save_folder, num_classes)
    if model is None:
        return  # Exit if loading fails

    st.success(f"Models loaded successfully! Ready to predict {num_classes} diseases.")

    # Sidebar configuration
    st.sidebar.title("🩺 Disease Predictor")
    st.sidebar.markdown("Enter symptoms to predict diseases and view treatments.")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920336.png", width=100)

    with st.sidebar.expander("Settings", expanded=False):
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=1.0, value=0.5, step=0.05,
            help="Minimum confidence for a valid prediction."
        )

    # Main content
    st.title("Disease Prediction with ClinicalBERT")
    st.markdown(
        """
        **Welcome!** Input symptoms to predict potential diseases and retrieve treatment suggestions.
        This app uses ClinicalBERT embeddings and a custom classifier trained on your dataset.
        """
    )

    # Symptom input
    st.subheader("Enter Symptoms")
    selected_symptoms = st.multiselect(
        "Select Common Symptoms",
        options=common_symptoms,
        default=["fever"],
        help="Choose from symptoms found in the dataset."
    )

    manual_symptoms = st.text_area(
        "Custom Symptoms (comma-separated)",
        placeholder="e.g., cough, fatigue",
        height=100,
        help="Add additional symptoms not in the dropdown."
    )

    symptoms = ", ".join(selected_symptoms + [s.strip() for s in manual_symptoms.split(",") if s.strip()])
    st.write(f"**Current Symptoms:** {symptoms if symptoms else 'None entered'}")

    if symptoms:
        st.info("Symptoms entered. Click 'Predict' to proceed.")
    else:
        st.warning("Please enter at least one symptom.")

    # Prediction button
    if st.button("Predict", key="predict_button"):
        if symptoms:
            progress_bar = st.progress(0)
            with st.spinner("Analyzing symptoms..."):
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                try:
                    disease, treatment, confidence = predict(
                        symptoms, tokenizer, clinicalbert_model, model, label_encoder, df, confidence_threshold
                    )

                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Diagnosis")
                        badge_color = "green" if confidence >= 0.75 else "orange" if confidence >= 0.5 else "red"
                        st.markdown(
                            f"**Predicted Disease:** {disease}  <span style='background-color:{badge_color};color:white;padding:2px 5px;border-radius:3px'>{confidence:.2%}</span>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Confidence:** {confidence:.2%}")
                    with col2:
                        st.subheader("Treatment")
                        st.write(f"**Suggested Treatment:** {treatment}")

                    # Confidence visualization
                    st.subheader("Prediction Confidence")
                    fig = px.bar(
                        x=["Predicted Disease"], y=[confidence],
                        labels={"x": "Prediction", "y": "Confidence"},
                        title="Confidence Level",
                        color=[confidence], color_continuous_scale="Blues",
                        range_y=[0, 1],
                        text=[f"{confidence:.2%}"]
                    )
                    fig.update_traces(textposition='auto')
                    st.plotly_chart(fig, use_container_width=True)

                    # Store in session state
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "Symptoms": symptoms,
                        "Disease": disease,
                        "Treatment": treatment,
                        "Confidence": f"{confidence:.2%}"
                    })

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            progress_bar.empty()
        else:
            st.error("No symptoms provided!")

    # Prediction history
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
        **Disclaimer:** This app uses ClinicalBERT for disease prediction and retrieves treatments from a dataset.
        Results are for demonstration purposes and should be validated by a healthcare professional.
        Built with Streamlit, PyTorch, and Transformers.
        """
    )

if __name__ == "__main__":
    main()
