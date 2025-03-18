# streamlit app 
## madakixo

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import pickle
import os
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Omdena Kaduna Disease Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model class (must match training script)
class OptimizedDiseaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OptimizedDiseaseClassifier, self).__init__()
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
        self.dropout_layers = [layer for layer in self.network if isinstance(layer, nn.Dropout)]

    def forward(self, x):
        return self.network(x)

    def update_dropout(self, p):
        for layer in self.dropout_layers:
            layer.p = p

# Load model and artifacts
@st.cache_resource
def load_model_and_artifacts(save_folder, num_classes):
    with open(os.path.join(save_folder, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    input_dim = len(tfidf.vocabulary_)

    model = OptimizedDiseaseClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    checkpoint = torch.load(os.path.join(save_folder, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(os.path.join(save_folder, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    faiss_index = faiss.read_index(os.path.join(save_folder, "faiss_index.bin"))

    return model, tfidf, label_encoder, faiss_index

# Load dataset and encode Disease column
@st.cache_data
def load_dataset_and_symptoms(data_path, _label_encoder):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    # Encode Disease column to match training
    df["Disease"] = _label_encoder.transform(df["Disease"])
    all_symptoms = set()
    for symptoms in df["Symptoms"]:
        all_symptoms.update([s.strip() for s in symptoms.split(",")])
    return df, sorted(list(all_symptoms))

# Prediction function with confidence
def predict(symptoms, model, tfidf, label_encoder, faiss_index, df):
    symptoms_tfidf = tfidf.transform([symptoms]).toarray()
    symptoms_tensor = torch.tensor(symptoms_tfidf, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(symptoms_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    disease_name = label_encoder.inverse_transform([predicted_class])[0]

    # Filter and handle empty result
    disease_rows = df[df["Disease"] == predicted_class]
    if disease_rows.empty:
        treatment = "Treatment information not available"
    else:
        disease_info = disease_rows.iloc[0]
        treatment = disease_info.get("Treatment", "Treatment information not available")

    symptoms_vec = symptoms_tfidf.astype(np.float32)
    _, indices = faiss_index.search(symptoms_vec, 1)
    if indices[0][0] >= len(df):
        similar_disease_name = "Unknown"
    else:
        similar_disease = df.iloc[indices[0][0]]["Disease"]
        similar_disease_name = label_encoder.inverse_transform([similar_disease])[0]

    top_k = 3
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_diseases = label_encoder.inverse_transform(top_indices.cpu().numpy())
    top_confidences = top_probs.cpu().numpy()

    return disease_name, treatment, similar_disease_name, confidence, top_diseases, top_confidences

# Main app
def main():
    # Define paths (update these to your local paths)
    save_folder = "model_classifier-002b"  # Adjust as needed
    data_path = "processed_diseases-priority.csv"  # Adjust as needed

    # Load data and model
    try:
        num_classes = len(pd.read_csv(data_path)["Disease"].unique())
        model, tfidf, label_encoder, faiss_index = load_model_and_artifacts(save_folder, num_classes)
        df_filtered, common_symptoms = load_dataset_and_symptoms(data_path, label_encoder)
        st.success(f"Model and artifacts loaded successfully! Input dim: {len(tfidf.vocabulary_)}")
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return

    # Session state for prediction history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar
    st.sidebar.title("🩺 NLP for Medical Prescription 2025")
    st.sidebar.markdown("Select or type symptoms to predict diseases.")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1053/1053171.png", width=100)

    # Main content
    st.title("Leveraging NLP for Disease Prediction")
    st.markdown("**Explore symptoms and get real-time disease predictions!**")

    # Symptom selection
    st.subheader("Select Symptoms")
    selected_symptoms = st.multiselect(
        "Choose common symptoms (or type below)",
        options=common_symptoms,
        default=["Fever", "Cough", "Nausea"]
    )

    # Manual symptom input
    manual_symptoms = st.text_area(
        "Additional Symptoms (comma-separated)",
        placeholder="e.g., fatigue, headache,",
        height=100,
        help="Add symptoms not in the list above."
    )

    # Combine symptoms
    symptoms = ", ".join(selected_symptoms + [s.strip() for s in manual_symptoms.split(",") if s.strip()])
    st.write(f"**Current Symptoms:** {symptoms if symptoms else 'None entered'}")

    # Real-time feedback
    if symptoms:
        st.info("Symptoms valid! Click 'Predict' to proceed.")
    else:
        st.warning("Please enter or select at least one symptom.")

    # Predict button
    if st.button("Predict", key="predict_button"):
        if symptoms:
            with st.spinner("Analyzing symptoms..."):
                try:
                    disease, treatment, similar_disease, confidence, top_diseases, top_confidences = predict(
                        symptoms, model, tfidf, label_encoder, faiss_index, df_filtered
                    )

                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prediction")
                        st.info(f"**Disease:** {disease}")
                        st.write(f"**Treatment:** {treatment}")
                        st.write(f"**Confidence:** {confidence:.2%}")

                    with col2:
                        st.subheader("Similar Condition")
                        st.warning(f"**Similar Disease:** {similar_disease}")
                        st.write("Consider this as a possible alternative.")

                    # Confidence visualization
                    st.subheader("Top Predictions")
                    fig = px.bar(
                        x=top_diseases,
                        y=top_confidences,
                        labels={"x": "Disease", "y": "Confidence"},
                        title="Top 3 Predicted Diseases",
                        color=top_confidences,
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Update history
                    st.session_state.history.append({
                        "Symptoms": symptoms,
                        "Disease": disease,
                        "Treatment": treatment,
                        "Confidence": f"{confidence:.2%}",
                        "Similar Disease": similar_disease
                    })

                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.error("No symptoms provided!")

    # Prediction history
    if st.session_state.history:
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
        **Note:** This app is for educational purposes only. Consult a healthcare professional for medical advice.
        Built with ❤️ using Streamlit, PyTorch, and Plotly Collaboratively on OmdenaAI.
        """
    )

if __name__ == "__main__":
    main()
