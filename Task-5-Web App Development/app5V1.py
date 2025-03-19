#streamlit app for use with torch_nn.py script in modeling dir
#madakixo

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
import time

# Set page configuration
st.set_page_config(
    page_title="Leveraging NLP in Medicial Prescription",
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
    df["Disease"] = _label_encoder.transform(df["Disease"])
    all_symptoms = set()
    for symptoms in df["Symptoms"]:
        all_symptoms.update([s.strip() for s in symptoms.split(",")])
    return df, sorted(list(all_symptoms))

# Prediction function with confidence
def predict(symptoms, model, tfidf, label_encoder, faiss_index, df, confidence_threshold=0.5):
    symptoms_tfidf = tfidf.transform([symptoms]).toarray()
    symptoms_tensor = torch.tensor(symptoms_tfidf, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(symptoms_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    disease_name = label_encoder.inverse_transform([predicted_class])[0]

    disease_rows = df[df["Disease"] == predicted_class]
    if disease_rows.empty or confidence < confidence_threshold:
        treatment = "Treatment information not available (low confidence or no match)"
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
    # Define paths
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

    # Session state for prediction history and feedback
    if "history" not in st.session_state:
        st.session_state.history = []
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False

    # Sidebar
    st.sidebar.title("🩺 Disease Predictor")
    st.sidebar.markdown("Select or type symptoms to predict diseases.")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1053/1053171.png", width=100)

    with st.sidebar.expander("Advanced Options", expanded=False):
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence required for a prediction to be considered valid."
        )

    # Main content
    st.title("Interactive Disease Prediction")
    st.markdown("**Explore symptoms and get real-time disease predictions!**")

    # Symptom selection
    st.subheader("Select Symptoms")
    selected_symptoms = st.multiselect(
        "Choose common symptoms (or type below)",
        options=common_symptoms,
        default=["Fever", "Cough"],
        help="Select from common symptoms or add custom ones below."
    )

    manual_symptoms = st.text_area(
        "Additional Symptoms (comma-separated)",
        placeholder="e.g., fatigue, headache",
        height=100,
        help="Enter additional symptoms not listed above, separated by commas."
    )

    symptoms = ", ".join(selected_symptoms + [s.strip() for s in manual_symptoms.split(",") if s.strip()])
    st.write(f"**Current Symptoms:** {symptoms if symptoms else 'None entered'}")

    if symptoms:
        st.info("Symptoms valid! Click 'Predict' to proceed.")
    else:
        st.warning("Please enter or select at least one symptom.")

    # Predict button with progress bar
    if st.button("Predict", key="predict_button"):
        if symptoms:
            progress_bar = st.progress(0)
            with st.spinner("Analyzing symptoms..."):
                for i in range(100):
                    time.sleep(0.01)  # Simulate processing time
                    progress_bar.progress(i + 1)
                try:
                    disease, treatment, similar_disease, confidence, top_diseases, top_confidences = predict(
                        symptoms, model, tfidf, label_encoder, faiss_index, df_filtered, confidence_threshold
                    )

                    # Display results with confidence badges using st.markdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prediction")
                        badge_color = "green" if confidence >= 0.75 else "orange" if confidence >= 0.5 else "red"
                        st.markdown(
                            f"**Disease:** {disease}  <span style='background-color:{badge_color};color:white;padding:2px 5px;border-radius:3px'>{confidence:.2%}</span>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Treatment:** {treatment}")

                    with col2:
                        st.subheader("Similar Condition")
                        st.warning(f"**Similar Disease:** {similar_disease}")
                        st.write("Consider this as a possible alternative.")

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

                    st.session_state.history.append({
                        "Symptoms": symptoms,
                        "Disease": disease,
                        "Treatment": treatment,
                        "Confidence": f"{confidence:.2%}",
                        "Similar Disease": similar_disease
                    })

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            progress_bar.empty()
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

    # Feedback section
    st.subheader("Provide Feedback")
    st.write("Help us improve! Share your thoughts:")
    
    with st.expander("Quick Feedback", expanded=False):
        feedback_text = st.text_area("Your Feedback", placeholder="How was your experience? Any suggestions?")
        if st.button("Submit Feedback") and feedback_text:
            st.session_state.feedback_submitted = True
            st.success("Thank you for your feedback!")
        if st.session_state.feedback_submitted:
            st.write("Feedback submitted!")

    google_form_link = "https://forms.gle/your-google-form-link"  # Replace with your Google Form URL
    st.markdown(f"For detailed feedback, please fill out [this form]({google_form_link}).")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Note:** This app is for educational purposes only. Consult a healthcare professional for medical advice.
        Built with ❤️ using Streamlit, PyTorch, and Plotly
        Project initiated by Jamaludeen Madaki for Omdena Kaduna Impact Hub
        Collaboratively executed by the Project Team.
        """
    )

if __name__ == "__main__":
    main()
