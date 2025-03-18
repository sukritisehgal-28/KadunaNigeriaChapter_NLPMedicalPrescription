# modeling task 4 
## @madakixo

# modeling task 4
## @madakixo
# Purpose: Train a neural network to classify diseases based on symptoms, using data augmentation,
# TF-IDF feature extraction, and FAISS for similarity search. Designed for Google Colab with Google Drive integration.

import pandas as pd  # For data manipulation and DataFrame operations
import numpy as np  # For numerical computations and array handling
import torch  # Core PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Neural network module in PyTorch for defining layers and loss functions
import torch.optim as optim  # Optimization algorithms (e.g., AdamW) for training
from sklearn.model_selection import train_test_split  # To split data into training and validation sets
from sklearn.preprocessing import LabelEncoder  # To encode disease names into numerical labels
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert symptom text into numerical features
from sklearn.metrics import f1_score  # To evaluate model performance using F1-score
from torch.utils.data import DataLoader, Dataset  # For batching and loading data into the model
import faiss  # For efficient similarity search using FAISS index
import pickle  # To save and load Python objects (e.g., TF-IDF vectorizer, label encoder)
import os  # For file and directory operations
from google.colab import drive  # To mount Google Drive in Colab for file access
import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import wordnet  # WordNet for synonym replacement in data augmentation
import random  # For random operations in augmentation

# Download required NLTK datasets for text augmentation
nltk.download('wordnet')  # Download WordNet for synonym lookup
nltk.download('averaged_perceptron_tagger')  # Download tagger (though not used here, included for potential future use)

# Mount Google Drive to access dataset and save artifacts
drive.mount('/content/drive')  # Connects Colab to Google Drive, prompting user authorization

# Define computation device: use GPU (CUDA) if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")  # Confirm device selection

# Define file paths for data and model artifacts
data_path = "/content/drive/MyDrive/projects2024/processed_diseases-priority.csv"  # Path to input CSV with symptoms, diseases, treatments
save_folder = "/content/drive/MyDrive/model/model_classifier-002b"  # Directory to save model and artifacts
os.makedirs(save_folder, exist_ok=True)  # Create save directory if it doesn’t exist, avoid errors if it does

# Load the dataset from CSV into a Pandas DataFrame
df = pd.read_csv(data_path)  # Read CSV containing columns like "Symptoms", "Disease", "Treatment"
print(f"✅ Dataset loaded: {df.shape[0]} records")  # Display number of records (rows)
print(f"Columns in original data: {list(df.columns)}")  # Display column names for verification

# Clean the dataset by removing rows with missing values
df.dropna(inplace=True)  # Drop rows with any NaN values to ensure data quality
print(f"✅ Data cleaned. Remaining records: {df.shape[0]}")  # Report remaining records after cleaning

# Define data augmentation functions to increase dataset size and diversity
def synonym_replacement(text, n=1):
    """Replace n words in text with synonyms using WordNet."""
    words = text.split()  # Split text into individual words
    new_words = words.copy()  # Create a copy to modify
    # Get unique words with synonyms available in WordNet
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)  # Shuffle for random selection
    num_replaced = 0  # Counter for replacements
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)  # Get synonym sets for the word
        if synonyms:  # If synonyms exist
            synonym = synonyms[0].lemmas()[0].name()  # Take first synonym from first synset
            if synonym != random_word:  # Ensure synonym differs from original
                new_words = [synonym if word == random_word else word for word in new_words]  # Replace word
                num_replaced += 1
        if num_replaced >= n:  # Stop after n replacements
            break
    return ' '.join(new_words)  # Rejoin words into a string

def random_deletion(text, p=0.2):
    """Randomly delete words from text with probability p."""
    words = text.split()  # Split text into words
    if len(words) <= 1:  # If one word or fewer, return unchanged
        return text
    new_words = [word for word in words if random.random() > p]  # Keep word if random > p
    return ' '.join(new_words) if new_words else random.choice(words)  # Return joined words or a random word if empty

# Apply data augmentation to triple the dataset size
augmented_data = []  # List to store augmented records
for idx, row in df.iterrows():  # Iterate over each row in the original DataFrame
    original_symptoms = row["Symptoms"]  # Extract original symptoms
    disease = row["Disease"]  # Extract disease name
    treatment = row.get("Treatment", "N/A")  # Extract treatment, default to "N/A" if missing
    # Add original record
    augmented_data.append({"Symptoms": original_symptoms, "Disease": disease, "Treatment": treatment})
    # Add record with synonym replacement
    augmented_data.append({"Symptoms": synonym_replacement(original_symptoms), "Disease": disease, "Treatment": treatment})
    # Add record with random deletion
    augmented_data.append({"Symptoms": random_deletion(original_symptoms), "Disease": disease, "Treatment": treatment})

df_augmented = pd.DataFrame(augmented_data)  # Convert list of dicts to DataFrame
print(f"✅ Augmented dataset size: {df_augmented.shape[0]} records")  # Report new dataset size
print(f"Columns in augmented data: {list(df_augmented.columns)}")  # Confirm columns remain consistent

# Filter out classes (diseases) with fewer than min_samples occurrences to ensure trainability
min_samples = 2  # Minimum samples per class for reliable training
class_counts = df_augmented["Disease"].value_counts()  # Count occurrences of each disease
valid_classes = class_counts[class_counts >= min_samples].index  # Identify diseases with enough samples
df_filtered = df_augmented[df_augmented["Disease"].isin(valid_classes)].copy()  # Filter DataFrame
removed_classes = len(class_counts) - len(valid_classes)  # Calculate number of removed classes
print(f"✅ Removed {removed_classes} classes with fewer than {min_samples} samples")  # Report removed classes
print(f"✅ Remaining records after filtering: {df_filtered.shape[0]}")  # Report remaining records
print(f"Columns in filtered data: {list(df_filtered.columns)}")  # Verify columns

# Encode disease labels into integers for classification
label_encoder = LabelEncoder()  # Initialize label encoder
df_filtered["Disease"] = label_encoder.fit_transform(df_filtered["Disease"])  # Transform disease names to integers
num_classes = len(label_encoder.classes_)  # Number of unique diseases
print(f"✅ Encoded {num_classes} unique diseases.")  # Report number of classes

# Extract features from symptoms using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')  # Initialize TF-IDF with max 5000 features, remove stop words
X = tfidf.fit_transform(df_filtered["Symptoms"]).toarray()  # Convert symptoms to TF-IDF matrix
y = df_filtered["Disease"].values  # Extract encoded labels as array
print(f"✅ Number of TF-IDF features: {X.shape[1]}")  # Report actual feature count (may be < 5000 if vocab is smaller)

# Compute class weights to handle imbalanced data
class_counts = np.bincount(y)  # Count occurrences of each class in y
class_weights = 1.0 / class_counts  # Inverse frequency for weighting (higher weight to rare classes)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Convert to PyTorch tensor
class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights to sum to num_classes

# Split data into training and validation sets, adjusting test size based on class count
test_size = max(0.2, num_classes / len(y))  # Ensure at least one sample per class in validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y  # Stratify to maintain class distribution
)
print(f"✅ Data split with test_size={test_size:.3f}")  # Report test size used

# Define a custom Dataset class for PyTorch DataLoader
class SymptomDataset(Dataset):
    """Custom Dataset to handle symptom features and labels."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert features to tensor
        self.y = torch.tensor(y, dtype=torch.long)  # Convert labels to tensor

    def __len__(self):
        return len(self.X)  # Return dataset size

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # Return single sample (features, label)

# Create datasets and data loaders
batch_size = min(32, len(X_train) // 4)  # Set batch size, capped at 32 or 1/4 of training size
train_dataset = SymptomDataset(X_train, y_train)  # Training dataset
val_dataset = SymptomDataset(X_val, y_val)  # Validation dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # Training loader with shuffling
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)  # Validation loader

# Define the neural network model with dropout scheduling
class OptimizedDiseaseClassifier(nn.Module):
    """Neural network for disease classification with batch norm and dropout."""
    def __init__(self, input_dim, num_classes):
        super(OptimizedDiseaseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),  # First layer: input_dim to 512 units
            nn.BatchNorm1d(512),  # Batch normalization for stability
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.4),  # Dropout with 40% probability
            nn.Linear(512, 256),  # Second layer: 512 to 256 units
            nn.BatchNorm1d(256),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.3),  # Dropout with 30% probability
            nn.Linear(256, num_classes)  # Output layer: 256 to num_classes
        )
        self.dropout_layers = [layer for layer in self.network if isinstance(layer, nn.Dropout)]  # Track dropout layers

    def forward(self, x):
        return self.network(x)  # Forward pass through the network

    def update_dropout(self, p):
        """Update dropout probability dynamically during training."""
        for layer in self.dropout_layers:
            layer.p = p  # Set new dropout probability

# Initialize model, loss, optimizer, and scheduler
model = OptimizedDiseaseClassifier(input_dim=X.shape[1], num_classes=num_classes).to(device)  # Create model instance
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Loss with class weights and smoothing
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)  # AdamW optimizer with regularization
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # Cosine annealing learning rate scheduler

# Define training function with F1-score evaluation and early stopping
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, patience=5):
    """Train the model with validation and early stopping based on F1-score."""
    best_val_f1 = 0.0  # Track best validation F1-score
    patience_counter = 0  # Counter for early stopping

    for epoch in range(epochs):  # Loop over epochs
        model.train()  # Set model to training mode
        running_loss = 0.0  # Accumulate loss
        correct = 0  # Track correct predictions
        total = 0  # Track total samples
        train_preds, train_labels = [], []  # Store predictions and labels
        
        dropout_p = max(0.1, 0.4 - (epoch * 0.015))  # Decrease dropout from 0.4 to 0.1 over epochs
        model.update_dropout(dropout_p)  # Update dropout probability

        for inputs, labels in train_loader:  # Iterate over training batches
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to prevent explosion
            optimizer.step()  # Update weights
            running_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total samples
            train_preds.extend(predicted.cpu().numpy())  # Store predictions
            train_labels.extend(labels.cpu().numpy())  # Store labels
        
        train_acc = correct / total  # Calculate training accuracy
        train_f1 = f1_score(train_labels, train_preds, average='weighted')  # Calculate weighted F1-score

        model.eval()  # Set model to evaluation mode
        val_correct = 0  # Track validation correct predictions
        val_total = 0  # Track validation total samples
        val_preds, val_labels = [], []  # Store validation predictions and labels
        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in val_loader:  # Iterate over validation batches
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
                outputs = model(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                val_correct += (predicted == labels).sum().item()  # Count correct predictions
                val_total += labels.size(0)  # Count total samples
                val_preds.extend(predicted.cpu().numpy())  # Store predictions
                val_labels.extend(labels.cpu().numpy())  # Store labels
        
        val_acc = val_correct / val_total  # Calculate validation accuracy
        val_f1 = f1_score(val_labels, val_preds, average='weighted')  # Calculate weighted F1-score

        # Print training and validation metrics
        print(f"✅ Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        scheduler.step()  # Update learning rate

        if val_f1 > best_val_f1:  # If validation F1 improves
            best_val_f1 = val_f1  # Update best F1-score
            torch.save({  # Save model checkpoint
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1
            }, os.path.join(save_folder, "best_model.pth"))
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
            if patience_counter >= patience:  # If no improvement for 'patience' epochs
                print(f"✅ Early stopping triggered after {patience} epochs without improvement")
                break  # Exit training loop

# Train the model
train(model, train_loader, val_loader, criterion, optimizer, scheduler)

# Build FAISS index for similarity search
faiss_index = faiss.IndexFlatL2(X.shape[1])  # Initialize FAISS index with L2 distance
faiss_index.add(np.array(X, dtype=np.float32))  # Add TF-IDF features to index
print("✅ FAISS index built")  # Confirm index creation

# Save model and artifacts
torch.save(model.state_dict(), os.path.join(save_folder, "final_model.pth"))  # Save final model state
with open(os.path.join(save_folder, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)  # Save TF-IDF vectorizer
with open(os.path.join(save_folder, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)  # Save label encoder
faiss.write_index(faiss_index, os.path.join(save_folder, "faiss_index.bin"))  # Save FAISS index
print(f"✅ Model and artifacts saved to {save_folder}")  # Confirm save operation

# Define functions to load artifacts and make predictions
def load_model_and_artifacts(save_folder, input_dim, num_classes):
    """Load trained model and artifacts for inference."""
    model = OptimizedDiseaseClassifier(input_dim=input_dim, num_classes=num_classes).to(device)  # Initialize model
    checkpoint = torch.load(os.path.join(save_folder, "best_model.pth"), map_location=device)  # Load best checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load weights
    model.eval()  # Set to evaluation mode

    with open(os.path.join(save_folder, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)  # Load TF-IDF vectorizer
    with open(os.path.join(save_folder, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)  # Load label encoder
    faiss_index = faiss.read_index(os.path.join(save_folder, "faiss_index.bin"))  # Load FAISS index

    return model, tfidf, label_encoder, faiss_index

def predict(symptoms, model, tfidf, label_encoder, faiss_index, df):
    """Predict disease from symptoms and find similar disease using FAISS."""
    symptoms_tfidf = tfidf.transform([symptoms]).toarray()  # Convert input symptoms to TF-IDF features
    symptoms_tensor = torch.tensor(symptoms_tfidf, dtype=torch.float32).to(device)  # Convert to tensor
    with torch.no_grad():  # No gradient computation for inference
        output = model(symptoms_tensor)  # Forward pass
        predicted_class = torch.argmax(output).item()  # Get predicted class index
    disease_name = label_encoder.inverse_transform([predicted_class])[0]  # Decode to disease name

    disease_info = df[df["Disease"] == predicted_class].iloc[0]  # Get disease info from DataFrame
    treatment = disease_info.get("Treatment", "Treatment information not available")  # Extract treatment

    symptoms_vec = symptoms_tfidf.astype(np.float32)  # Convert to float32 for FAISS
    _, indices = faiss_index.search(symptoms_vec, 1)  # Search for 1 nearest neighbor
    similar_disease = df.iloc[indices[0][0]]["Disease"]  # Get similar disease index
    similar_disease_name = label_encoder.inverse_transform([similar_disease])[0]  # Decode to name

    return disease_name, treatment, similar_disease_name  # Return prediction and similar disease

# Test the prediction function
loaded_model, loaded_tfidf, loaded_label_encoder, loaded_faiss_index = load_model_and_artifacts(save_folder, X.shape[1], num_classes)
example_symptoms = "Fever, cough, fatigue"  # Example input symptoms
predicted_disease, predicted_treatment, similar_disease = predict(
    example_symptoms, loaded_model, loaded_tfidf, loaded_label_encoder, loaded_faiss_index, df_filtered
)
print(f"🔹 Symptoms: {example_symptoms}")  # Display input symptoms
print(f"🔹 Predicted Disease: {predicted_disease}")  # Display predicted disease
print(f"🔹 Treatment: {predicted_treatment}")  # Display treatment
print(f"🔹 Similar Disease (FAISS): {similar_disease}")  # Display similar disease
