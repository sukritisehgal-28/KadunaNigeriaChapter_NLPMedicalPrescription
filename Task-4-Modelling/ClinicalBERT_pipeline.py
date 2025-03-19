#script uses ClinicalBERT to predict treatment and diseases from symptoms
#script  produces model limited by dataset

#madakixo 

import pandas as pd  # For data manipulation and DataFrame operations
import numpy as np  # For numerical computations and array handling
import torch  # Core PyTorch library for tensors and neural networks
import torch.nn as nn  # Neural network module for layers and loss
import torch.optim as optim  # Optimization algorithms (e.g., AdamW)
from sklearn.model_selection import train_test_split  # Split data into train/validation
from sklearn.preprocessing import LabelEncoder  # Encode disease labels
from sklearn.metrics import f1_score  # Evaluate model with F1-score
from torch.utils.data import DataLoader, Dataset  # For batching and data loading
import pickle  # To save/load model artifacts
import os  # For file and directory operations
from google.colab import drive  # Mount Google Drive in Colab
import nltk  # Natural Language Toolkit for text augmentation
from nltk.corpus import wordnet  # WordNet for synonym replacement
import random  # For random operations in augmentation
from transformers import AutoTokenizer, AutoModel  # Load ClinicalBERT tokenizer and model

# Download NLTK data for augmentation
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Mount Google Drive
drive.mount('/content/drive')

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Define paths
data_path = "/content/drive/MyDrive/bioGPT/processed_diseases-priority.csv"
save_folder = "/content/drive/MyDrive/bioGPT/models/model_classifier-clinicalbert"
os.makedirs(save_folder, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)
print(f"✅ Dataset loaded: {df.shape[0]} records")
print(f"Columns in original data: {list(df.columns)}")

# Clean Data
df.dropna(inplace=True)
print(f"✅ Data cleaned. Remaining records: {df.shape[0]}")

# Data Augmentation Functions
def synonym_replacement(text, n=1):
    """Replace n words with synonyms using WordNet."""
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            if synonym != random_word:
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def random_deletion(text, p=0.2):
    """Randomly delete words with probability p."""
    words = text.split()
    if len(words) <= 1:
        return text
    new_words = [word for word in words if random.random() > p]
    return ' '.join(new_words) if new_words else random.choice(words)

# Apply Data Augmentation
augmented_data = []
for idx, row in df.iterrows():
    original_symptoms = row["Symptoms"]
    disease = row["Disease"]
    treatment = row.get("Treatment", "N/A")
    augmented_data.append({"Symptoms": original_symptoms, "Disease": disease, "Treatment": treatment})
    augmented_data.append({"Symptoms": synonym_replacement(original_symptoms), "Disease": disease, "Treatment": treatment})
    augmented_data.append({"Symptoms": random_deletion(original_symptoms), "Disease": disease, "Treatment": treatment})

df_augmented = pd.DataFrame(augmented_data)
print(f"✅ Augmented dataset size: {df_augmented.shape[0]} records")
print(f"Columns in augmented data: {list(df_augmented.columns)}")

# Filter out classes with insufficient samples
min_samples = 2
class_counts = df_augmented["Disease"].value_counts()
valid_classes = class_counts[class_counts >= min_samples].index
df_filtered = df_augmented[df_augmented["Disease"].isin(valid_classes)].copy()
removed_classes = len(class_counts) - len(valid_classes)
print(f"✅ Removed {removed_classes} classes with fewer than {min_samples} samples")
print(f"✅ Remaining records after filtering: {df_filtered.shape[0]}")
print(f"Columns in filtered data: {list(df_filtered.columns)}")

# Encode Labels
label_encoder = LabelEncoder()
df_filtered["Disease"] = label_encoder.fit_transform(df_filtered["Disease"])
num_classes = len(label_encoder.classes_)
print(f"✅ Encoded {num_classes} unique diseases.")

# Load ClinicalBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinicalbert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
clinicalbert_model.eval()  # Set to evaluation mode for feature extraction

# Function to extract ClinicalBERT embeddings
def get_clinicalbert_embeddings(texts, max_length=128):
    """Extract ClinicalBERT embeddings for a list of symptom texts."""
    embeddings = []
    with torch.no_grad():  # No gradient computation for inference
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to(device)
            outputs = clinicalbert_model(**inputs)
            # Use mean of token embeddings (CLS token could also be used: outputs.last_hidden_state[:, 0, :])
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

# Extract embeddings for symptoms
X = get_clinicalbert_embeddings(df_filtered["Symptoms"].values)
y = df_filtered["Disease"].values
print(f"✅ ClinicalBERT embeddings extracted: {X.shape}")  # Should be (num_samples, 768)

# Compute class weights for imbalanced data
class_counts = np.bincount(y)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
class_weights = class_weights / class_weights.sum() * num_classes  # Normalize

# Split data
test_size = max(0.2, num_classes / len(y))
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)
print(f"✅ Data split with test_size={test_size:.3f}")

# Define Dataset Class
class SymptomDataset(Dataset):
    """Custom Dataset for ClinicalBERT embeddings and labels."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
batch_size = min(32, len(X_train) // 4)
train_dataset = SymptomDataset(X_train, y_train)
val_dataset = SymptomDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

# Define Enhanced Model
class ClinicalBERTDiseaseClassifier(nn.Module):
    """Neural network for disease classification using ClinicalBERT embeddings."""
    def __init__(self, input_dim=768, num_classes=num_classes):  # ClinicalBERT outputs 768-dim embeddings
        super(ClinicalBERTDiseaseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),  # Input is 768 from ClinicalBERT
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

# Initialize model, loss, optimizer, scheduler
model = ClinicalBERTDiseaseClassifier(input_dim=768, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, patience=5):
    """Train the model with validation and early stopping based on F1-score."""
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_preds, train_labels = [], []

        dropout_p = max(0.1, 0.4 - (epoch * 0.015))  # Dropout scheduling
        model.update_dropout(dropout_p)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = correct / total
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        model.eval()
        val_correct = 0
        val_total = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"✅ Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1
            }, os.path.join(save_folder, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"✅ Early stopping triggered after {patience} epochs without improvement")
                break

# Train the model
train(model, train_loader, val_loader, criterion, optimizer, scheduler)

# Save artifacts (excluding FAISS for simplicity)
torch.save(model.state_dict(), os.path.join(save_folder, "final_model.pth"))
with open(os.path.join(save_folder, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)
print(f"✅ Model and label encoder saved to {save_folder}")

# Load and Predict Functions
def load_model_and_artifacts(save_folder, num_classes):
    """Load trained model and label encoder (ClinicalBERT is loaded separately for inference)."""
    model = ClinicalBERTDiseaseClassifier(input_dim=768, num_classes=num_classes).to(device)
    checkpoint = torch.load(os.path.join(save_folder, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(os.path.join(save_folder, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder

def predict(symptoms, model, label_encoder, df):
    """Predict disease and treatment from symptoms using ClinicalBERT embeddings."""
    # Extract ClinicalBERT embedding for input symptoms
    with torch.no_grad():
        inputs = tokenizer(symptoms, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
        outputs = clinicalbert_model(**inputs)
        symptoms_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # Mean pooling

    symptoms_tensor = symptoms_embedding.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(symptoms_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    disease_name = label_encoder.inverse_transform([predicted_class])[0]
    disease_info = df[df["Disease"] == predicted_class].iloc[0]
    treatment = disease_info.get("Treatment", "Treatment information not available")

    return disease_name, treatment

# Test Prediction
loaded_model, loaded_label_encoder = load_model_and_artifacts(save_folder, num_classes)
example_symptoms = "Fever, cough, fatigue"
predicted_disease, predicted_treatment = predict(example_symptoms, loaded_model, loaded_label_encoder, df_filtered)
print(f"🔹 Symptoms: {example_symptoms}")
print(f"🔹 Predicted Disease: {predicted_disease}")
print(f"🔹 Treatment: {predicted_treatment}")
