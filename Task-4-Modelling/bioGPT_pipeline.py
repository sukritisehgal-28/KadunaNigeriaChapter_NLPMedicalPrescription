#script to train model useing ClinicalBERT and BioGPT 
#for use with app7V1.py in Task 5 web development 
# madakixo
#install sacremoses for bioGPT access after installing transformer & datasets

#bioGPT for dynamic prediction generation

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
import pickle
import os
from google.colab import drive
import nltk
from nltk.corpus import wordnet
import random
from transformers import AutoTokenizer, AutoModel, BioGptTokenizer, BioGptForCausalLM

# Download NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Mount Google Drive
drive.mount('/content/drive')

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Define paths
data_path = "/content/drive/MyDrive/bioGPT/processed_diseases-priority.csv"
save_folder = "/content/drive/MyDrive/bioGPT/models/model_classifier-clinicalbert-biogpt"
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

# Filter out classes with insufficient samples
min_samples = 2
class_counts = df_augmented["Disease"].value_counts()
valid_classes = class_counts[class_counts >= min_samples].index
df_filtered = df_augmented[df_augmented["Disease"].isin(valid_classes)].copy()
removed_classes = len(class_counts) - len(valid_classes)
print(f"✅ Removed {removed_classes} classes with fewer than {min_samples} samples")
print(f"✅ Remaining records after filtering: {df_filtered.shape[0]}")

# Encode Labels
label_encoder = LabelEncoder()
df_filtered["Disease"] = label_encoder.fit_transform(df_filtered["Disease"])
num_classes = len(label_encoder.classes_)
print(f"✅ Encoded {num_classes} unique diseases.")

# Load ClinicalBERT tokenizer and model
clinicalbert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinicalbert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
clinicalbert_model.eval()

# Load BioGPT tokenizer and model for treatment generation
biogpt_tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large")
biogpt_model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large").to(device)
biogpt_model.eval()

# Extract ClinicalBERT embeddings
def get_clinicalbert_embeddings(texts, max_length=128):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = clinicalbert_tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to(device)
            outputs = clinicalbert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

X = get_clinicalbert_embeddings(df_filtered["Symptoms"].values)
y = df_filtered["Disease"].values
print(f"✅ ClinicalBERT embeddings extracted: {X.shape}")

# Compute class weights
class_counts = np.bincount(y)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
class_weights = class_weights / class_weights.sum() * num_classes

# Split data
test_size = max(0.2, num_classes / len(y))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
print(f"✅ Data split with test_size={test_size:.3f}")

# Dataset Class
class SymptomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoaders
batch_size = min(32, len(X_train) // 4)
train_dataset = SymptomDataset(X_train, y_train)
val_dataset = SymptomDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

# Model Definition
class ClinicalBERTDiseaseClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=num_classes):
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
        self.dropout_layers = [layer for layer in self.network if isinstance(layer, nn.Dropout)]

    def forward(self, x):
        return self.network(x)

    def update_dropout(self, p):
        for layer in self.dropout_layers:
            layer.p = p

# Initialize model
model = ClinicalBERTDiseaseClassifier(input_dim=768, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Training Function
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, patience=5):
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_preds, train_labels = [], []

        dropout_p = max(0.1, 0.4 - (epoch * 0.015))
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

# Save artifacts
torch.save(model.state_dict(), os.path.join(save_folder, "final_model.pth"))
with open(os.path.join(save_folder, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)
print(f"✅ Model and label encoder saved to {save_folder}")

# Load and Predict Functions
def load_model_and_artifacts(save_folder, num_classes):
    model = ClinicalBERTDiseaseClassifier(input_dim=768, num_classes=num_classes).to(device)
    checkpoint = torch.load(os.path.join(save_folder, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(os.path.join(save_folder, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder

def predict(symptoms, model, label_encoder, biogpt_model, biogpt_tokenizer):
    # Predict disease with ClinicalBERT
    with torch.no_grad():
        inputs = clinicalbert_tokenizer(symptoms, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
        outputs = clinicalbert_model(**inputs)
        symptoms_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    symptoms_tensor = symptoms_embedding.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(symptoms_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    disease_name = label_encoder.inverse_transform([predicted_class])[0]

    # Generate treatment with BioGPT
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

    return disease_name, treatment

# Test Prediction
loaded_model, loaded_label_encoder = load_model_and_artifacts(save_folder, num_classes)
example_symptoms = "Fever, cough, fatigue"
predicted_disease, predicted_treatment = predict(example_symptoms, loaded_model, loaded_label_encoder, biogpt_model, biogpt_tokenizer)
print(f"🔹 Symptoms: {example_symptoms}")
print(f"🔹 Predicted Disease: {predicted_disease}")
print(f"🔹 Generated Treatment: {predicted_treatment}")

